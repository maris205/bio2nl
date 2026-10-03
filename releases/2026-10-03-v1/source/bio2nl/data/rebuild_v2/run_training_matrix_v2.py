"""Run the accepted v2 pretraining matrix serially on one local GPU."""
import datetime as dt
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[3]
RELEASE = WORKSPACE / "data_rebuild/2026-09-25-v2"
RUNS = WORKSPACE / "bio2nl/review/retrain_v2_2026-09-25"
SEED = 0


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_status(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gate = json.loads((RELEASE / "validation/integrated_release_acceptance.json").read_text())
    if not gate.get("training_gate_pass"):
        raise SystemExit("The integrated v2 training gate is not passed.")
    matrix = json.loads((RELEASE / "configs/training_matrix_v2.json").read_text())
    smallweb = json.loads((RELEASE / "configs/smallweb_gpt2_v2.json").read_text())
    seeds = matrix.get("pretraining_seeds", [])
    if seeds != [SEED] or smallweb.get("seed") != SEED:
        raise SystemExit("Queue seed does not match the frozen v2 run configuration.")

    tasks = []
    for tokenizer, conditions in (("pure_byte", matrix["conditions"]["bytelevel"]),
                                  ("mixed_bpe", matrix["conditions"]["mixed_bpe"])):
        for condition in conditions:
            for size in ("tiny", "small"):
                name = f"{tokenizer}__{condition}__{size}__seed{SEED}"
                output = RUNS / "models" / name
                command = [sys.executable, str(WORKSPACE / "bio2nl/data/rebuild_v2/train_from_scratch_v2.py"),
                           "--root", str(RELEASE), "--tokenizer", tokenizer, "--condition", condition,
                           "--size", size, "--seed", str(SEED), "--output", str(output)]
                tasks.append({"name": name, "command": command, "output": str(output)})
    name = f"gpt2_smallweb__seed{SEED}"
    output = RUNS / "models" / name
    command = [sys.executable, str(WORKSPACE / "bio2nl/data/rebuild_v2/train_smallweb_gpt2_v2.py"),
               "--root", str(RELEASE), "--seed", str(SEED), "--output", str(output)]
    tasks.append({"name": name, "command": command, "output": str(output)})

    RUNS.mkdir(parents=True, exist_ok=True)
    logdir = RUNS / "logs"
    logdir.mkdir(exist_ok=True)
    plan = {"release_id": "2026-09-25-v2", "seed": SEED, "device": "local GPU 0",
            "created_at": now(), "status": "queued", "tasks": [
                {"name": t["name"], "command": t["command"], "output": t["output"], "status": "pending"}
                for t in tasks]}
    status_path = RUNS / "queue_status.json"
    if args.dry_run:
        plan["status"] = "dry_run"
        write_status(status_path, plan)
        print(json.dumps({"status": plan["status"], "task_count": len(tasks),
                          "tasks": [t["name"] for t in tasks], "status_file": str(status_path)}, indent=2))
        return

    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    used_mib, total_mib = (int(x.strip()) for x in gpu.split(","))
    if used_mib >= 500:
        raise SystemExit(f"GPU 0 is not free ({used_mib}/{total_mib} MiB in use); queue not started.")
    write_status(status_path, plan)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false")
    for i, task in enumerate(tasks):
        output = Path(task["output"])
        if output.exists():
            plan["status"] = "stopped_existing_output"
            plan["tasks"][i]["status"] = "blocked_output_exists"
            plan["tasks"][i]["finished_at"] = now()
            write_status(status_path, plan)
            raise SystemExit(f"Refusing to overwrite existing output: {output}")
        logpath = logdir / f"{task['name']}.log"
        plan["status"] = "running"
        plan["current_task"] = task["name"]
        plan["tasks"][i].update(status="running", started_at=now(), log=str(logpath))
        write_status(status_path, plan)
        with logpath.open("w") as log:
            log.write("command: " + " ".join(task["command"]) + "\nstarted_at: " + now() + "\n")
            log.flush()
            result = subprocess.run(task["command"], cwd=WORKSPACE, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        plan["tasks"][i].update(status="completed" if result.returncode == 0 else "failed",
                                finished_at=now(), returncode=result.returncode)
        if result.returncode != 0:
            plan["status"] = "stopped_after_failure"
            plan["failed_task"] = task["name"]
            write_status(status_path, plan)
            raise SystemExit(f"Training stopped at {task['name']} (exit {result.returncode}); see {logpath}")
        plan["completed_tasks"] = i + 1
        write_status(status_path, plan)

    plan["status"] = "completed"
    plan["current_task"] = None
    plan["finished_at"] = now()
    write_status(status_path, plan)
    print(json.dumps({"status": plan["status"], "completed_tasks": len(tasks),
                      "status_file": str(status_path)}, indent=2))


if __name__ == "__main__":
    main()
