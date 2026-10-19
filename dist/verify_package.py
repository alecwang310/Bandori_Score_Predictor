#!/usr/bin/env python3
"""Regenerate or verify dist/PACKAGE_SHA256.json over the distribution contents.

    python3 verify_package.py            # verify every recorded file (default)
    python3 verify_package.py --regen    # regenerate the manifest, then verify

Excluded from the manifest: PACKAGE_SHA256.json itself, cache/, output/,
__pycache__/ and *.pyc (runtime scratch), .venv/.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED_DIRS = {'cache', 'output', '__pycache__', '.venv'}
EXCLUDED_NAMES = {'PACKAGE_SHA256.json'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dist_files():
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.name in EXCLUDED_NAMES or path.suffix == '.pyc':
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        yield rel


def main(argv):
    manifest_path = ROOT / 'PACKAGE_SHA256.json'
    if '--regen' in argv:
        files = {rel.as_posix(): sha256(ROOT / rel) for rel in dist_files()}
        manifest_path.write_text(json.dumps(files, indent=1, sort_keys=True) + '\n')
        print(f'regenerated {manifest_path.name}: {len(files)} files')
    if not manifest_path.exists():
        print('FAIL: PACKAGE_SHA256.json missing (run with --regen)', file=sys.stderr)
        return 1
    recorded = json.loads(manifest_path.read_text())
    problems = []
    for name in sorted(recorded):
        path = ROOT / name
        if not path.exists():
            problems.append('missing file: ' + name)
        elif sha256(path) != recorded[name]:
            problems.append('sha256 mismatch: ' + name)
    current = {rel.as_posix() for rel in dist_files()}
    for name in sorted(current - set(recorded)):
        problems.append('file not in manifest (rerun --regen): ' + name)
    if problems:
        for line in problems:
            print('FAIL:', line, file=sys.stderr)
        return 1
    print(f'OK: {len(recorded)} files match PACKAGE_SHA256.json')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
