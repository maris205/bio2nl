"""Integrated data acceptance, code snapshot, and non-circular SHA-256 manifest.

Returns a staging manifest with explicit gates when any required component is
missing. A release is never marked accepted merely because source files exist.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import digest, write_json


def copy_code(workspace, root):
    copies={
      "biopaws/data_v2":workspace/"biopaws/data_v2",
      "biopaws/data_v2_remote":workspace/"biopaws/data_v2_remote",
      "bio2nl/data/rebuild_v2":workspace/"bio2nl/data/rebuild_v2",
    }
    code=root/"code";code.mkdir(exist_ok=True)
    for target,source in copies.items():
        dest=code/target
        if dest.exists():shutil.rmtree(dest)
        shutil.copytree(source,dest,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for project in ("bio2nl","biopaws"):
        sha=subprocess.check_output(["git","-C",str(workspace/project),"rev-parse","HEAD"],text=True).strip()
        status=subprocess.check_output(["git","-C",str(workspace/project),"status","--porcelain"],text=True)
        write_json(root/"metadata"/f"code_snapshot_{project}.json",dict(repository=project,base_commit=sha,
            workspace_dirty=bool(status),snapshot_path=f"code/{project}",
            note="Exact code snapshot files are listed by the release manifest; base commit is not asserted to contain the uncommitted snapshot."))


def check_tokenized(root):
    issues=[]; summaries=[]
    expectations={"pure_byte":260,"mixed_bpe":None}
    for family in ("pure_byte","mixed_bpe"):
        tok_path=root/"tokenizers"/family/"tokenizer.json"
        if not tok_path.exists():issues.append(f"missing tokenizer {family}");continue
        tokenizer=json.loads(tok_path.read_text());vocab=tokenizer["model"]["vocab"]
        if expectations[family] is not None and len(vocab)!=expectations[family]:issues.append(f"{family} vocabulary size {len(vocab)}")
        metas={}
        for condition in ("english","protein","shuffled","randomaa","dna"):
            for split,target in (("train",16_777_216),("validation",262_144),("test",262_144)):
                meta=root/"tokenized"/family/condition/(split+".json")
                idx=root/"tokenized"/family/condition/(split+".index.jsonl")
                binary=root/"tokenized"/family/condition/(split+".bin")
                if not (meta.exists() and idx.exists() and binary.exists()):
                    issues.append(f"missing {family}/{condition}/{split}");continue
                row=json.loads(meta.read_text());size=binary.stat().st_size
                if row["tokens"]!=target or size!=target*2:issues.append(f"budget/bytes {family}/{condition}/{split}")
                offset=0
                with idx.open() as stream:
                    for line in stream:
                        entry=json.loads(line)
                        if entry["offset"]!=offset or entry["length"]<1:
                            issues.append(f"broken record offsets {family}/{condition}/{split}");break
                        offset+=entry["length"]
                if offset!=target:issues.append(f"index end {family}/{condition}/{split}: {offset}")
                summaries.append({"family":family,"condition":condition,"split":split,"tokens":row["tokens"],"sha256":row["sha256"]})
    gpt_meta=root/"tokenizers/gpt2_pinned/metadata.json"
    if not gpt_meta.exists():
        issues.append("missing pinned GPT-2 tokenizer snapshot")
    else:
        for split,target in (("train",49_999_872),("validation",262_144),("test",262_144)):
            meta=root/"tokenized/gpt2_smallweb"/(split+".json")
            idx=root/"tokenized/gpt2_smallweb"/(split+".index.jsonl")
            binary=root/"tokenized/gpt2_smallweb"/(split+".bin")
            if not (meta.exists() and idx.exists() and binary.exists()):
                issues.append(f"missing GPT2 SmallWeb/{split}");continue
            row=json.loads(meta.read_text());size=binary.stat().st_size
            if row["tokens"]!=target or size!=target*2:issues.append(f"GPT2 budget/bytes {split}")
            offset=0
            with idx.open() as stream:
                for line in stream:
                    entry=json.loads(line)
                    if entry["offset"]!=offset or entry["length"]<1:
                        issues.append(f"broken GPT2 record offsets {split}");break
                    offset+=entry["length"]
            if offset!=target:issues.append(f"GPT2 index end {split}: {offset}")
            summaries.append({"family":"gpt2_pinned","condition":"english","split":split,"tokens":row["tokens"],"sha256":row["sha256"]})
    return {"pass":not issues,"issues":issues,"bins":summaries}


def gates(root):
    result={"nlp_official":False,"nlp_clean":False,"typed_dyck":False,"longrange_synthetic":False,"english_corpus":False,
            "dna_corpus":False,"protein_pairs":False,"remote_pairs":False,"protein_pretraining_exclusion":False,
            "cross_dataset_homology_diagnostic":False,"all_tokenizers_and_bins":False}
    issues=[]
    checks=[("nlp_official",root/"validation/nlp_acceptance_report.json","construction_acceptance_pass"),
            ("nlp_clean",root/"validation/nlp_clean_acceptance_report.json","acceptance_pass"),
            ("typed_dyck",root/"validation/synthetic_dyck_acceptance_report.json","acceptance_pass"),
            ("english_corpus",root/"validation/corpora_english.json","condition"),
            ("dna_corpus",root/"validation/corpora_dna.json","condition"),
            ("protein_pairs",root/"validation/protein_pair_acceptance.json","acceptance_pass"),
            ("remote_pairs",root/"validation/remote_validation.json","status"),
            ("protein_pretraining_exclusion",root/"metadata/protein_remote_pretraining_exclusion.json","status"),
            ("cross_dataset_homology_diagnostic",root/"validation/cross_dataset_homology.json","status")]
    for key,path,field in checks:
        if not path.exists():continue
        try:d=json.loads(path.read_text())
        except Exception as ex:issues.append(f"invalid JSON {path}: {ex}");continue
        if key in ("nlp_official","nlp_clean","typed_dyck","protein_pairs"):
            result[key]=d.get(field) is True
        elif key in ("english_corpus","dna_corpus"):
            result[key]=d.get("condition")=="english" if key=="english_corpus" else d.get("condition")=="dna"
            result[key]=result[key] and not any(d.get("exact_text_cross_split_overlap",{}).values())
        elif key=="remote_pairs":result[key]=d.get(field)=="passed_local_data_checks"
        else:result[key]=d.get(field)=="completed"
    tokenized=check_tokenized(root);result["all_tokenizers_and_bins"]=tokenized["pass"]
    issues.extend(tokenized["issues"])
    longrange=[root/"validation"/f"synthetic_longrange_d{n}_acceptance.json" for n in (8,16,24)]
    if all(p.exists() for p in longrange):
        reports=[json.loads(p.read_text()) for p in longrange]
        result["longrange_synthetic"]=all(x.get("acceptance_pass") is True and all(abs(v["majority"]-.5)<1e-12 and abs(v["nearest_noun_agreement"]-.5)<1e-12 for v in x["baselines_accuracy"].values()) for x in reports)
    else:issues.append("one or more rebuilt longrange synthetic tasks are missing")
    return result,issues,tokenized


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--workspace",type=Path,default=Path.cwd());a=p.parse_args();root=a.root.resolve();workspace=a.workspace.resolve()
    copy_code(workspace,root)
    gates_result,issues,tokenized=gates(root)
    required=all(gates_result.values())
    write_json(root/"validation/integrated_release_acceptance.json",{"status":"accepted_for_training" if required and not issues else "staging_blocked",
      "training_gate_pass":required and not issues,"gates":gates_result,"issues":issues,"tokenized_bins":tokenized["bins"],
      "limitation":"A passed gate certifies only its enumerated rules; it does not prove all dataset leakage or biological shortcuts absent."})
    excluded_dirs={"work","logs","__pycache__"}
    excluded_suffixes={".partial",".pyc",".tmp",".log"}
    files=[]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded_dirs for part in path.relative_to(root).parts):continue
        if path.suffix in excluded_suffixes or path.name in {"manifest.json","SHA256SUMS"}:continue
        files.append({"file":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":digest(path)})
    manifest={"release_id":"2026-09-25-v2","status":"accepted_for_training" if required and not issues else "staging",
        "training_gate_pass":required and not issues,"gates":gates_result,"integrated_acceptance":"validation/integrated_release_acceptance.json",
        "file_count":len(files),"total_bytes":sum(x["bytes"] for x in files),"files":files,
        "manifest_policy":"This manifest excludes itself and SHA256SUMS; code snapshot files and provenance are included. Work caches and partial downloads are omitted."}
    write_json(root/"manifest.json",manifest)
    (root/"SHA256SUMS").write_text("".join(f"{x['sha256']}  {x['file']}\n" for x in files))
    print(json.dumps({"status":manifest["status"],"gates":gates_result,"issues":issues,"files":len(files),"bytes":manifest["total_bytes"]},indent=2),flush=True)


if __name__=="__main__":main()
