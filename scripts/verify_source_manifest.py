#!/usr/bin/env python3
"""Verify the pinned Git checkout and per-file SHA-256 manifest."""
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream-mind'
MANIFEST = ROOT / 'source_manifest.json'
EXPECTED = 'f632bdf7b1136dff142835f9fe69c8ce5f8c166d'


def run(*args):
    return subprocess.check_output(args, cwd=UPSTREAM, text=True).strip()


def main():
    data = json.loads(MANIFEST.read_text())
    head = run('git', 'rev-parse', 'HEAD')
    tree = run('git', 'rev-parse', 'HEAD^{tree}')
    status = run('git', 'status', '--porcelain')
    if head != EXPECTED or data['commit'] != EXPECTED:
        raise SystemExit(f'commit mismatch: HEAD={head}, expected={EXPECTED}')
    if tree != data['tree']:
        raise SystemExit(f'tree mismatch: {tree} != {data["tree"]}')
    if status:
        raise SystemExit(f'working tree is not clean:\n{status}')

    errors = []
    for item in data['files']:
        path = UPSTREAM / item['path']
        try:
            blob = subprocess.check_output(['git', 'cat-file', 'blob', f'HEAD:{item["path"]}'], cwd=UPSTREAM)
        except subprocess.CalledProcessError:
            errors.append(f'missing Git blob: {item["path"]}')
            continue
        digest = hashlib.sha256(blob).hexdigest()
        mode = run('git', 'ls-tree', 'HEAD', '--', item['path']).split()[0]
        if len(blob) != item['bytes'] or digest != item['sha256'] or mode != item['mode']:
            errors.append(f'manifest mismatch: {item["path"]}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'OK: commit={head}; tree={tree}; files={len(data["files"])}; SHA-256 and modes match')


if __name__ == '__main__':
    main()
