"""Portable GPT-2 models for the frozen joint-pretraining experiment.

No workspace paths, legacy module imports, downloads or device initialization are
performed at import time. Checkpoints are accepted only with independently trusted
file, tensor-state, job and provenance identities. The caller supplies those
identities from the verified archive; a hash read from the checkpoint itself is
not an independent trust anchor.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Invalid " + name)
    return value


def state_digest(state):
    """Exact legacy name/dtype/shape framing, including all tied aliases."""
    digest = hashlib.sha256()
    require(isinstance(state, Mapping) and bool(state), "Empty or invalid tensor state")
    require(all(isinstance(name, str) and torch.is_tensor(value)
                for name, value in state.items()), "Invalid tensor-state entry")
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        data = value.numpy().tobytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def snapshot_state(model):
    """Copy finite FP32 tensors to CPU while retaining exact shared aliases."""
    aliases, result = {}, {}
    for name, tensor in model.state_dict().items():
        require(tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(),
                "Invalid/nonfinite FP32 state: " + name)
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
        if key not in aliases:
            aliases[key] = tensor.detach().cpu().clone().contiguous()
        result[name] = aliases[key]
    return result


def _seed(seed):
    require(type(seed) is int and seed >= 0, "Seed must be a nonnegative integer")
    return seed


def make_config(design):
    """Construct the offline architecture described by the frozen protocol."""
    pretraining, tokenizer = design["pretraining"], design["tokenizer"]
    values = dict(pretraining["architecture"])
    require(values.pop("model_type", "gpt2") == "gpt2", "Only GPT-2 is supported")
    config = GPT2Config(**values, vocab_size=tokenizer["vocab_size"],
                        n_positions=pretraining["context_length"], n_ctx=pretraining["context_length"],
                        bos_token_id=tokenizer["eos_token_id"], eos_token_id=tokenizer["eos_token_id"],
                        pad_token_id=tokenizer["pad_token_id"])
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return config


def create_pretraining_model(design, seed):
    """Create a fresh FP32 LM without consuming the caller's CPU/GPU RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(_seed(seed))
        return GPT2LMHeadModel(make_config(design)).float()


class TransferClassifier(nn.Module):
    """Original backbone/score state names and mask-based final-token pooling."""

    def __init__(self, backbone, head_seed):
        super().__init__()
        if isinstance(backbone, GPT2LMHeadModel):
            backbone = backbone.transformer
        if not isinstance(backbone, GPT2Model):
            raise TypeError("Expected GPT2Model or GPT2LMHeadModel")
        self.backbone = backbone
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_seed(head_seed))
            self.score = nn.Linear(backbone.config.n_embd, 2, bias=False, device="cpu", dtype=torch.float32)
            nn.init.normal_(self.score.weight, std=0.02)
        reference = next(backbone.parameters())
        self.score.to(device=reference.device, dtype=reference.dtype)

    def forward(self, input_ids, attention_mask):
        require(input_ids.ndim == 2 and input_ids.shape == attention_mask.shape,
                "Input/mask dimensions differ")
        require(input_ids.shape[0] > 0 and input_ids.shape[1] > 0 and
                torch.all((attention_mask == 0) | (attention_mask == 1)).item() and
                torch.all(attention_mask.sum(1) > 0).item(), "Invalid or empty attention mask")
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        last = positions.masked_fill(attention_mask == 0, -1).amax(dim=1)
        return self.score(hidden[torch.arange(input_ids.shape[0], device=input_ids.device), last])


def create_classifier(lm, fine_tuning_seed):
    """Reuse the complete LM backbone and initialize only the two-logit head."""
    require(isinstance(lm, GPT2LMHeadModel), "SFT must start from a complete GPT-2 LM")
    return TransferClassifier(lm, fine_tuning_seed).float()


def load_checkpoint(path, *, expected_sha256, expected_kind, expected_job,
                    expected_state_sha256, expected_protocol_sha256, expected_manifest_sha256,
                    expected_epoch=None, expected_config=None):
    """Safely load a trusted frozen checkpoint on CPU with exact provenance.

    The file SHA is checked on the same open file handle used for weights-only
    deserialization. There is deliberately no unsafe pickle fallback. Returned
    models are in evaluation mode with gradients disabled; training from a new
    initialization uses the constructors above.
    """
    for value, name in ((expected_sha256, "file SHA-256"),
                        (expected_state_sha256, "state SHA-256"),
                        (expected_protocol_sha256, "protocol SHA-256"),
                        (expected_manifest_sha256, "manifest SHA-256")):
        _sha256(value, name)
    require(expected_kind in ("causal_lm", "source_classifier"), "Invalid expected checkpoint kind")
    require(isinstance(expected_job, dict) and bool(expected_job), "Expected job identity is required")
    require(expected_epoch is None or (type(expected_epoch) is int and expected_epoch > 0),
            "Invalid expected epoch")
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        require(digest.hexdigest() == expected_sha256, "Checkpoint file SHA-256 differs")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and type(payload.get("schema_version")) is int and
            payload["schema_version"] == 1 and payload.get("kind") == expected_kind,
            "Invalid checkpoint schema/kind")
    metadata, config_values, state = payload.get("metadata"), payload.get("config"), payload.get("state_dict")
    require(isinstance(metadata, dict) and isinstance(config_values, dict), "Invalid checkpoint metadata/config")
    require(metadata.get("job") == expected_job, "Checkpoint job differs")
    require(metadata.get("protocol_sha256") == expected_protocol_sha256, "Checkpoint protocol differs")
    require(metadata.get("prepared_manifest_sha256") == expected_manifest_sha256, "Checkpoint manifest differs")
    require(expected_epoch is None or metadata.get("epoch") == expected_epoch, "Checkpoint epoch differs")
    require(payload.get("state_sha256") == expected_state_sha256, "Checkpoint declared state differs")
    require(isinstance(state, Mapping) and bool(state) and
            all(isinstance(name, str) and torch.is_tensor(value) and value.dtype == torch.float32
                and torch.isfinite(value).all().item() for name, value in state.items()),
            "Invalid/nonfinite FP32 checkpoint state")
    require(state_digest(state) == expected_state_sha256, "Checkpoint tensor-state SHA-256 differs")
    require(config_values.get("model_type", "gpt2") == "gpt2", "Invalid checkpoint architecture")
    if expected_config is not None:
        require(isinstance(expected_config, Mapping) and bool(expected_config), "Invalid expected config")
        require(all(config_values.get(key) == value for key, value in expected_config.items()),
                "Checkpoint config differs")
    config = GPT2Config.from_dict(config_values)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    with torch.random.fork_rng(devices=[]):
        if expected_kind == "causal_lm":
            model = GPT2LMHeadModel(config)
        else:
            model = TransferClassifier(GPT2Model(config), 0)
        model.load_state_dict(state, strict=True)
    require(state_digest(model.state_dict()) == expected_state_sha256, "Reloaded model state differs")
    model.float().eval().requires_grad_(False)
    return model, payload


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def validate_arrays(arrays, config):
    """Reject inconsistent encodings without silently changing any input row."""
    require(isinstance(arrays, Mapping) and {"input_ids", "attention_mask", "labels"} <= arrays.keys(),
            "Missing encoded arrays")
    values = {name: _as_numpy(arrays[name]) for name in ("input_ids", "attention_mask", "labels")}
    ids, mask, labels = (values[name] for name in ("input_ids", "attention_mask", "labels"))
    require(all(value.dtype == np.int64 for value in values.values()), "Encoded arrays must be int64")
    require(ids.ndim == 2 and ids.shape == mask.shape and labels.shape == (len(ids),) and
            len(ids) > 0 and 0 < ids.shape[1] <= config.n_positions, "Invalid encoded array dimensions")
    require(np.all((ids >= 0) & (ids < config.vocab_size)), "Token ID outside vocabulary")
    require(np.all((mask == 0) | (mask == 1)) and np.all(mask.sum(1) > 0) and
            np.all(np.diff(mask, axis=1) <= 0), "Expected nonempty right-padded binary masks")
    require(np.all((labels == 0) | (labels == 1)), "Expected binary labels")
    return values


@torch.inference_mode()
def predict_source(model, arrays, batch_size=32, device="cpu", precision="float32"):
    """Fixed-head inference; label 1 direction and argmax ties to label 0."""
    require(isinstance(model, TransferClassifier), "Expected a complete source classifier")
    require(type(batch_size) is int and batch_size > 0, "Invalid batch size")
    device = torch.device(device)
    require((device.type == "cpu" and precision == "float32") or
            (device.type == "cuda" and precision == "bfloat16"), "Explicit CPU FP32 or CUDA BF16 required")
    if device.type == "cuda":
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 unavailable")
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                "TF32 must be disabled for fixed scoring")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    require(all(parameter.device == device and parameter.dtype == torch.float32
                for parameter in model.parameters()), "Model must already be FP32 on the explicit device")
    values = validate_arrays(arrays, model.backbone.config)
    model.eval()
    chunks = []
    for start in range(0, len(values["labels"]), batch_size):
        stop = min(start + batch_size, len(values["labels"]))
        batch_mask = values["attention_mask"][start:stop]
        maximum = int(batch_mask.sum(1).max())
        width = min(batch_mask.shape[1], ((maximum + 7) // 8) * 8)
        ids = torch.as_tensor(values["input_ids"][start:stop, :width], device=device)
        mask = torch.as_tensor(batch_mask[:, :width], device=device)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context:
            logits = model(ids, mask)
        logp = F.log_softmax(logits.float(), dim=-1)
        require(torch.isfinite(logp).all().item(), "Nonfinite source predictions")
        chunks.append(logp.cpu().numpy().astype(np.float64))
    logp = np.concatenate(chunks)
    return {"labels": values["labels"].copy(), "log_probabilities": logp, "predictions": logp.argmax(1)}
