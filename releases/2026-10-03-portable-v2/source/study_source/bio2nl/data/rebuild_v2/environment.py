"""Write a compact environment record for a data rebuild release."""
import importlib.metadata, json, platform, sys
from pathlib import Path


def collect():
    packages={}
    for name in ("pyarrow","tokenizers","numpy","scipy","biopython","scikit-learn","pandas","requests","transformers","huggingface-hub"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return {"python":sys.version,"python_implementation":platform.python_implementation(),
            "platform":platform.platform(),"machine":platform.machine(),"packages":packages,
            "data_pipeline_uses_gpu":False}


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    path=a.root/"environment/data_build_environment.json";path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(collect(),indent=2)+"\n")
    req=a.root/"environment/data_build_requirements.txt"
    req.write_text("\n".join(f"{n}=={v}" for n,v in collect()["packages"].items() if v)+"\n")
    print(path)
