"""Write the two-file patch overlay only; never run construction or acceptance."""
import argparse
import json
from pathlib import Path

from .patches import write_patched_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = write_patched_snapshot(args.source_root, args.output)
    print(json.dumps({'status': result['status'], 'files': len(result['files'])}))


if __name__ == '__main__':
    main()
