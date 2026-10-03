import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "protein_phase.py"
SPEC = importlib.util.spec_from_file_location("protein_phase", MODULE)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class ProteinPhaseTests(unittest.TestCase):
    def test_plan_uses_only_accepted_stages_and_explicit_roots(self):
        plan = p.stage_plan(Path("/new release"), Path("/snapshot"), "/python")
        self.assertEqual([s["name"] for s in plan], ["normalize", "cluster", "search", "graph_split",
                         "remote_prepare", "remote_exclusion", "protein_pairs", "remote_pairs", "cross_dataset_audit"])
        for stage in plan:
            self.assertIn("/new release", stage["command"])
            self.assertNotIn("screen_remote.py", " ".join(stage["command"]))
            self.assertNotIn("finalize_release.py", " ".join(stage["command"]))

    def test_inventory_hashes_content_and_rejects_missing_or_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").write_text("old")
            first = p.inventory(root, ["a"])["a"]["sha256"]
            (root / "a").write_text("new")
            self.assertNotEqual(first, p.inventory(root, ["a"])["a"]["sha256"])
            with self.assertRaises(FileNotFoundError):
                p.inventory(root, ["missing"])
            (root / "link").symlink_to(root / "a")
            with self.assertRaises(ValueError):
                p.inventory(root, ["link"])

    def test_stop_first_failure_and_cpu_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"; logs.mkdir()
            (root / "script.py").write_text("pass\n")
            (root / "input").write_text("raw")
            stages = [{"name": name, "script": "script.py", "command": ["stub", name],
                       "inputs": ["input"], "outputs": [name]} for name in ("one", "two", "three")]
            calls = []
            def execute(command, **kwargs):
                calls.append(command[-1])
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                self.assertNotIn("PYTHONPATH", kwargs["env"])
                kwargs["stdout"].write(command[-1] + "\n")
                if command[-1] == "one": (root / "one").write_text("result")
                return subprocess.CompletedProcess(command, 0 if command[-1] == "one" else 7)
            with patch.dict("os.environ", {"PYTHONPATH": "/injected"}):
                with self.assertRaisesRegex(RuntimeError, "returned 7"):
                    p.execute_stages(root, root, logs, stages, run_command=execute)
            self.assertEqual(calls, ["one", "two"])
            status = json.loads((logs / "protein_phase_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stages"][0]["status"], "completed")
            self.assertEqual(status["stages"][1]["returncode"], 7)
            self.assertEqual(status["stages"][0]["outputs"]["one"]["sha256"], p.digest(root / "one"))

    def test_zero_exit_with_missing_output_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); logs = root / "logs"; logs.mkdir()
            (root / "script.py").write_text("pass\n")
            stage = {"name": "missing", "script": "script.py", "command": ["stub"], "inputs": ["script.py"], "outputs": ["missing"]}
            def execute(command, **kwargs): return subprocess.CompletedProcess(command, 0)
            with self.assertRaises(FileNotFoundError):
                p.execute_stages(root, root, logs, [stage], run_command=execute)
            self.assertEqual(json.loads((logs / "protein_phase_status.json").read_text())["status"], "failed")

    def test_preflight_rejects_changed_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); builders = root / "builders"; builders.mkdir()
            key = next(iter(p.BUILDERS)); path = builders / key
            path.parent.mkdir(parents=True); path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                p.preflight(root, builders, root / "logs", [])

    def test_preflight_never_overwrites_existing_protein_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); builders = root / "builders"; builders.mkdir()
            path = root / "data/sequences"; path.mkdir(parents=True)
            (path / "already_built").write_text("preserve")
            with patch.object(p, "BUILDERS", {}), patch.object(p, "inventory", return_value={}):
                with self.assertRaises(FileExistsError):
                    p.preflight(root, builders, root / "logs", [])
            self.assertEqual((path / "already_built").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
