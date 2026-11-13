#!/usr/bin/env python3
"""Create a source-only Kaggle runtime bundle for the 15-field training notebook.

The archive deliberately excludes raw data, SQLite databases, vocabulary tables,
embeddings, checkpoints, virtual environments, and Kaggle credentials.
"""
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "mind_15field_runtime_source.zip"
MANIFEST_OUTPUT = Path(__file__).resolve().parent / "MIND_RUNTIME_SOURCE_MANIFEST.json"
EXPECTED_COMMIT = "f632bdf7b1136dff142835f9fe69c8ce5f8c166d"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_files():
    files = []
    for folder in ("mind_data_pipeline", "mind_news_metadata", "configs"):
        files.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".json"})
    files.append(ROOT / "mind_config.py")
    for name in ("source_lock.json", "source_manifest.json", "source_provenance.json"):
        path = ROOT / name
        if path.is_file():
            files.append(path)
    unique = {p.relative_to(ROOT).as_posix(): p for p in files}
    return dict(sorted(unique.items()))


def main():
    config = json.loads((ROOT / "configs/mind_no_body_15field.json").read_text(encoding="utf-8"))
    if config.get("source", {}).get("commit") != EXPECTED_COMMIT:
        raise ValueError("canonical config does not point to the locked upstream commit")

    paths = source_files()
    hashes = {name: sha256(path.read_bytes()) for name, path in paths.items()}
    payload_hash = sha256(json.dumps(hashes, sort_keys=True).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "bundle_type": "mind_15field_runtime_source_only",
        "upstream_commit": EXPECTED_COMMIT,
        "variant": config["variant"],
        "payload_sha256": payload_hash,
        "files": [
            {"path": name, "bytes": paths[name].stat().st_size, "sha256": hashes[name]}
            for name in paths
        ],
        "excluded": ["raw MIND", "SQLite/database", "vocabulary artifacts", "tokenizer files",
                     "embedding matrices", "checkpoints", "credentials", "virtual environments"],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in paths.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("MIND_RUNTIME_SOURCE_MANIFEST.json", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_text)
    # A root-level copy lets the Kaggle notebook identify this source-only
    # dataset before extracting the archive. The archive contains the same
    # manifest and is independently verified by the notebook.
    MANIFEST_OUTPUT.write_text(manifest_text, encoding="utf-8")

    print(json.dumps({
        "output": str(OUTPUT),
        "manifest_output": str(MANIFEST_OUTPUT),
        "upstream_commit": EXPECTED_COMMIT,
        "source_file_count": len(paths),
        "payload_sha256": payload_hash,
        "archive_bytes": OUTPUT.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
