"""Integrity/adversarial tests; no large models, GPU or training required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import GPT2Config, GPT2Model

import inference as api
import portable_core as core


HERE = Path(__file__).resolve().parent


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        cfg = GPT2Config(vocab_size=37, n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=2,
                         n_inner=32, eos_token_id=1, pad_token_id=0)
        cfg._attn_implementation = "sdpa"
        model = core.TransferClassifier(GPT2Model(cfg), 0)
        self.payload = {"schema_version": 1, "kind": "source_classifier", "config": cfg.to_dict(),
                        "metadata": {"job": {"condition": "EP", "pt_seed": 0, "ft_seed": 0, "smoke": False},
                                     "protocol_sha256": "a" * 64, "prepared_manifest_sha256": "b" * 64, "epoch": 2},
                        "state_dict": core.snapshot_state(model)}
        self.path = self.root / "toy.pt"
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.payload["state_sha256"] = core.state_digest(self.payload["state_dict"])
        torch.save(self.payload, self.path)
        self.kw = dict(expected_sha256=api.file_sha(self.path), expected_kind="source_classifier",
            expected_job=self.payload["metadata"]["job"], expected_state_sha256=self.payload["state_sha256"],
            expected_protocol_sha256="a" * 64, expected_manifest_sha256="b" * 64, expected_epoch=2,
            expected_config=self.payload["config"])

    def test_intact_reloads(self):
        model, payload = core.load_checkpoint(self.path, **self.kw)
        self.assertEqual(core.state_digest(model.state_dict()), payload["state_sha256"])
        self.assertFalse(model.training)
        self.assertTrue(all(not x.requires_grad for x in model.parameters()))

    def test_corrupt_file_rejected(self):
        with self.path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "file SHA"):
            core.load_checkpoint(self.path, **self.kw)

    def test_missing_head_rejected_even_with_new_digest(self):
        del self.payload["state_dict"]["score.weight"]
        self.save()
        with self.assertRaises(RuntimeError):
            core.load_checkpoint(self.path, **self.kw)

    def test_wrong_head_shape_rejected_even_with_new_digest(self):
        self.payload["state_dict"]["score.weight"] = torch.zeros(3, 16)
        self.save()
        with self.assertRaises(RuntimeError):
            core.load_checkpoint(self.path, **self.kw)

    def test_provenance_rejected(self):
        for key in ("expected_protocol_sha256", "expected_manifest_sha256", "expected_state_sha256"):
            changed = dict(self.kw, **{key: "c" * 64})
            with self.assertRaises(ValueError):
                core.load_checkpoint(self.path, **changed)

    def test_wrong_job_rejected(self):
        changed = dict(self.kw, expected_job={**self.kw["expected_job"], "ft_seed": 1})
        with self.assertRaisesRegex(ValueError, "job differs"):
            core.load_checkpoint(self.path, **changed)

    def test_wrong_epoch_rejected(self):
        with self.assertRaisesRegex(ValueError, "epoch differs"):
            core.load_checkpoint(self.path, **{**self.kw, "expected_epoch": 1})

    def test_wrong_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "config differs"):
            core.load_checkpoint(self.path, **{**self.kw, "expected_config": {"n_embd": 32}})

    def test_json_config_label_keys_roundtrip(self):
        public_config = json.loads(json.dumps(self.payload["config"]))
        digest = api.canonical_sha(public_config)
        expected = api.config_for_core(public_config)
        model, payload = core.load_checkpoint(self.path, **{**self.kw, "expected_config": expected})
        self.assertEqual(api.original_config_sha_after_load(payload["config"]), digest)
        self.assertEqual(api.canonical_sha(public_config), digest)

    def test_unexpected_config_runtime_value_rejected(self):
        changed = {**self.payload["config"], "attn_implementation": "eager"}
        with self.assertRaisesRegex(ValueError, "injected attention"):
            api.original_config_sha_after_load(changed)

    def test_nonfinite_rejected(self):
        self.payload["state_dict"]["score.weight"][0, 0] = float("nan")
        self.save()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            core.load_checkpoint(self.path, **self.kw)

    def test_path_escape_and_ambiguity_rejected(self):
        for relative in ("../secret", "/tmp/file", "a/../b", "a//b", "./a", "a\\b"):
            with self.assertRaises(ValueError):
                api.safe_path(self.root, relative)

    def test_catalog_file_tamper(self):
        path = HERE / "model_catalog.json"
        original = path.read_bytes()
        bad = self.root / "catalog.json"
        bad.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "Catalog SHA"):
            api.read_catalog(bad, api.file_sha(path))

    def test_wrong_parent_rejected_with_rehashed_catalog(self):
        data = json.loads((HERE / "model_catalog.json").read_text())
        entry = next(x for x in data["models"] if x["kind"] == "source_classifier")
        entry["pretraining_parent"]["id"] = "ES-pt0"
        path = self.root / "catalog.json"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "parent provenance"):
            api.read_catalog(path, api.file_sha(path))

    def test_tokenizer_hash_tamper(self):
        catalog = json.loads((HERE / "model_catalog.json").read_text())
        raw = (HERE / "tokenizer.json").read_bytes()
        (self.root / "tokenizer.json").write_bytes(raw.replace(b"BPE", b"XYZ", 1))
        with self.assertRaisesRegex(ValueError, "SHA"):
            api.load_tokenizer(catalog, self.root)

    def test_encoding_caps_and_roundtrip(self):
        catalog = json.loads((HERE / "model_catalog.json").read_text())
        tokenizer = api.load_tokenizer(catalog, HERE)
        pair = {"sentence1": "hello world " * 400, "sentence2": "protein " * 400}
        arrays, info = api.encode_pairs(tokenizer, [pair])
        a = tokenizer.encode(pair["sentence1"], add_special_tokens=False).ids[:255]
        b = tokenizer.encode(pair["sentence2"], add_special_tokens=False).ids[:255]
        self.assertEqual(arrays["input_ids"][0].tolist(), a + [2] + b + [1])
        self.assertTrue(arrays["attention_mask"].all())
        self.assertEqual(info[0]["truncated_endpoints"], [True, True])

    def test_labels_or_empty_endpoints_rejected(self):
        tokenizer = api.load_tokenizer(json.loads((HERE / "model_catalog.json").read_text()), HERE)
        for pair in ({"sentence1": "a", "sentence2": "b", "label": 1}, {"sentence1": "", "sentence2": "b"}):
            with self.assertRaises(ValueError):
                api.encode_pairs(tokenizer, [pair])


if __name__ == "__main__":
    unittest.main()
