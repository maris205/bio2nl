"""Classifier, role guards, and source-only selection for direct pair transfer.

No filesystem reads or device initialization occur here. Input roles are explicit:
only source train rows can supply gradients; target rows are evaluation-only.
"""
from __future__ import annotations

import hashlib
import operator
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Model


def integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def require_role(domain, split, purpose):
    allowed = {
        "gradient": {("source", "train")},
        "selection": {("source", "validation")},
        "final": {("source", "test"), ("target", "validation"), ("target", "test")},
    }
    if purpose not in allowed or (domain, split) not in allowed[purpose]:
        raise ValueError(f"Forbidden data role: {domain}/{split} for {purpose}")


class TransferClassifier(nn.Module):
    """One transformer and one head, unchanged between protein and English."""

    def __init__(self, backbone, head_seed):
        super().__init__()
        if isinstance(backbone, GPT2LMHeadModel):
            backbone = backbone.transformer
        if not isinstance(backbone, GPT2Model):
            raise TypeError("Expected GPT2Model or GPT2LMHeadModel")
        self.backbone = backbone
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(integer(head_seed, "head_seed"))
            self.score = nn.Linear(backbone.config.n_embd, 2, bias=False, device="cpu", dtype=torch.float32)
            nn.init.normal_(self.score.weight, std=.02)
        reference = next(backbone.parameters())
        self.score.to(device=reference.device, dtype=reference.dtype)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
        # Mask, rather than token-ID inequality, preserves the genuine final EOS.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        last = positions.masked_fill(attention_mask == 0, -1).amax(dim=1)
        return self.score(hidden[torch.arange(input_ids.shape[0], device=input_ids.device), last])


class TensorRows:
    """Role-checked unpadded rows, padded once and optionally cached on GPU."""

    def __init__(self, rows: Sequence[Mapping], *, domain, split, purpose,
                 pad=0, eos=1, separator=2, vocab_size=32000, max_length=512, device="cpu"):
        require_role(domain, split, purpose)
        self.domain, self.split, self.purpose = domain, split, purpose
        if not rows:
            raise ValueError("At least one nonempty row is required")
        sequences, labels, row_ids = [], [], []
        for row in rows:
            if row.get("domain") != domain or row.get("split") != split:
                raise ValueError("Row domain/split does not match requested access role")
            row_id = row["row_id"]
            if not isinstance(row_id, str) or not row_id:
                raise ValueError("Nonempty string row_id required")
            tokens = [integer(x, "token ID") for x in row["input_ids"]]
            if not 4 <= len(tokens) <= max_length:
                raise ValueError("Empty or invalid encoded pair")
            if min(tokens) < 0 or max(tokens) >= vocab_size:
                raise ValueError("Token outside vocabulary")
            if tokens[-1] != eos or tokens.count(eos) != 1 or tokens.count(separator) != 1 or pad in tokens:
                raise ValueError("Expected unpadded pair with one separator and one final EOS")
            boundary = tokens.index(separator)
            if not 1 <= boundary <= 255 or not 1 <= len(tokens) - boundary - 2 <= 255:
                raise ValueError("Each endpoint must contain 1..255 tokens")
            label = integer(row["label"], "label")
            if label not in (0, 1):
                raise ValueError("Binary label required")
            sequences.append(tokens)
            labels.append(label)
            row_ids.append(row_id)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Duplicate row IDs")
        self.row_ids = tuple(row_ids)
        self.lengths_cpu = torch.tensor([len(s) for s in sequences], dtype=torch.int64)
        self.labels_cpu = torch.tensor(labels, dtype=torch.int64)
        width = ((max(map(len, sequences)) + 7) // 8) * 8
        self.input_ids = torch.full((len(rows), width), pad, dtype=torch.int64)
        for i, seq in enumerate(sequences):
            self.input_ids[i, :len(seq)] = torch.tensor(seq)
        self.attention_mask = torch.arange(width).unsqueeze(0) < self.lengths_cpu.unsqueeze(1)
        self.labels = self.labels_cpu
        self.device = torch.device("cpu")
        self.to(device)

    def __len__(self):
        return len(self.row_ids)

    def to(self, device):
        device = torch.device(device)
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self

    def batch(self, indices):
        if isinstance(indices, torch.Tensor) and indices.device.type != "cpu":
            raise ValueError("Batch indices must reside on CPU")
        indices = torch.tensor([integer(i, "batch index") for i in indices], dtype=torch.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("Nonempty 1-D batch required")
        if int(indices.min()) < 0 or int(indices.max()) >= len(self):
            raise IndexError("Batch index out of range")
        width = ((int(self.lengths_cpu.index_select(0, indices).max()) + 7) // 8) * 8
        index = indices.to(self.device)
        return (self.input_ids[:, :width].index_select(0, index),
                self.attention_mask[:, :width].index_select(0, index),
                self.labels.index_select(0, index), tuple(self.row_ids[i] for i in indices.tolist()))


def classification_scores(labels, logp):
    labels, logp = np.asarray(labels, dtype=np.int64), np.asarray(logp, dtype=np.float64)
    if len(labels) == 0 or logp.shape != (len(labels), 2) or not np.isfinite(logp).all():
        raise ValueError("Malformed or nonfinite predictions")
    if not np.isin(labels, [0, 1]).all() or not np.allclose(np.exp(logp).sum(1), 1., atol=2e-6):
        raise ValueError("Invalid binary labels/probabilities")
    pred = logp.argmax(1)
    return {"rows": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "mcc": float(matthews_corrcoef(labels, pred)),
            # Positive means label 1 in both domains. Never flip using labels.
            "auroc": float(roc_auc_score(labels, logp[:, 1] - logp[:, 0])) if len(set(labels)) == 2 else None,
            "cross_entropy": float(-logp[np.arange(len(labels)), labels].mean()),
            "predicted_positive_fraction": float(pred.mean()),
            "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


def best_source_epoch(history):
    if not history:
        raise ValueError("No source validation history")
    for entry in history:
        value = entry["source_validation"]["metrics"]["cross_entropy"]
        if not np.isfinite(value) or int(entry["epoch"]) <= 0:
            raise ValueError("Invalid source validation candidate")
    return min(history, key=lambda h: (h["source_validation"]["metrics"]["cross_entropy"], h["epoch"]))


def select_source_pilot(results):
    """No target fields are inspected; ties prefer smaller LR then earlier epoch."""
    if len(results) != 2 or len({r["job"]["condition"] for r in results}) != 1:
        raise ValueError("Exactly two same-condition pilot results required")
    if {r["job"]["learning_rate"] for r in results} != {2e-5, 1e-4}:
        raise ValueError("Unexpected pilot learning rates")
    for result in results:
        if result["job"]["seed"] != 0 or result["job"]["role"] != "pilot":
            raise ValueError("Selection requires seed-zero source pilots")
        best = best_source_epoch(result["history"])
        if result["best_epoch"] != best["epoch"] or result["best_source_validation"] != best["source_validation"]:
            raise ValueError("Best source checkpoint disagrees with history")
    return min(results, key=lambda r: (r["best_source_validation"]["metrics"]["cross_entropy"],
                                     r["job"]["learning_rate"], r["best_epoch"]))


def head_digest(state):
    return hashlib.sha256(state["score.weight"].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
