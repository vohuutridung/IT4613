"""Build a current-vocabulary-aligned entity matrix from provided MIND vectors.

This is a deterministic, source-guided reimplementation of upstream's
merge-entity-emb.py: train -> dev -> test, first occurrence wins, and each
provided vector is L2-normalized. It is not the author's prebuilt matrix.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mind_config import ARTIFACT_CONFIG, ENCODER_CONFIG, PREPROCESSING_CONFIG


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "test")
SPECIAL = {"<PAD>", "<UNK>"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_vocab(path):
    tokens = []
    for line in open(path, encoding="utf-8"):
        idx, token, *_ = line.rstrip("\n").split("\t")
        idx = int(idx)
        if idx != len(tokens):
            raise ValueError("entity vocab IDs are not contiguous at {}".format(idx))
        tokens.append(token)
    if tokens[ENCODER_CONFIG.pad_id] != "<PAD>" or tokens[ENCODER_CONFIG.unk_id] != "<UNK>":
        raise ValueError("entity mapping must reserve PAD=0 and UNK=1")
    return tokens


def build(db_path, out_path, manifest_path, mapping_path, seed=None):
    seed = ARTIFACT_CONFIG.initialization_seed if seed is None else int(seed)
    db_path, out_path = Path(db_path), Path(out_path)
    manifest_path, mapping_path = Path(manifest_path), Path(mapping_path)
    tokens = _read_vocab(mapping_path)
    token_to_id = {token: i for i, token in enumerate(tokens)}
    if len(token_to_id) != len(tokens):
        raise ValueError("duplicate token in entity mapping")

    data_manifest_path = ROOT / "artifacts/audit/data_manifest.json"
    data_manifest = json.load(open(data_manifest_path, encoding="utf-8"))
    rng = np.random.RandomState(int(seed))
    width = int(ENCODER_CONFIG.embedding_dim)
    matrix = rng.uniform(-0.05, 0.05, (len(tokens), width)).astype(np.float32)
    seen = {}
    duplicate_rows = 0
    conflicting_ids = set()
    ignored_oov_rows = 0
    input_rows = {split: 0 for split in SPLITS}
    zero_norm_rows = 0

    for split in SPLITS:
        path = ROOT / "data" / ("MINDlarge_" + split) / "entity_embedding.vec"
        expected = data_manifest["splits"]["MINDlarge_" + split]["files"]["entity_embedding.vec"]
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError("raw entity vector input differs from retained manifest: {}".format(path))
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                fields = line.split()
                if len(fields) != width + 1:
                    raise ValueError("expected entity ID + {} values at {}:{}".format(width, path, line_no))
                entity = fields[0]
                values = np.asarray([float(x) for x in fields[1:]], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("non-finite entity vector at {}:{}".format(path, line_no))
                input_rows[split] += 1
                if entity in seen:
                    duplicate_rows += 1
                    if not np.array_equal(values, seen[entity]):
                        conflicting_ids.add(entity)
                    continue
                seen[entity] = values.copy()
                row = token_to_id.get(entity)
                if row is None:
                    ignored_oov_rows += 1
                    continue
                norm = float(np.linalg.norm(values))
                if norm == 0.0:
                    zero_norm_rows += 1
                    values = np.zeros_like(values)
                else:
                    values = values / norm
                matrix[row] = values.astype(np.float32)

    if conflicting_ids:
        # Keep upstream first-seen behavior, but make the conflict visible.
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix, allow_pickle=False)
    covered = set(seen).intersection(token_to_id)
    covered.discard("<PAD>")
    covered.discard("<UNK>")
    regular = [x for x in tokens if x not in SPECIAL]
    manifest = {
      "status": "DERIVED_FROM_PROVIDED_MIND_ENTITY_VECTORS_NOT_AUTHOR_PREBUILT_MATRIX",
      "source_commit": PREPROCESSING_CONFIG.source_commit,
      "method": "train->dev->test; first occurrence wins; per-vector L2 normalization, matching upstream src/tools/merge-entity-emb.py",
      "upstream_source_file": "upstream-mind/projects/ai/mind/src/tools/merge-entity-emb.py",
      "input_files": {
        split: {"path": "data/MINDlarge_" + split + "/entity_embedding.vec",
               "bytes": (ROOT / "data" / ("MINDlarge_" + split) / "entity_embedding.vec").stat().st_size,
               "sha256": sha256(ROOT / "data" / ("MINDlarge_" + split) / "entity_embedding.vec"),
               "rows": input_rows[split]}
        for split in SPLITS
      },
      "precedence": list(SPLITS),
      "duplicate_rows_after_first": duplicate_rows,
      "conflicting_entity_ids": sorted(conflicting_ids),
      "conflicting_entity_id_count": len(conflicting_ids),
      "out_of_vocab_unique_vectors_ignored": ignored_oov_rows,
      "coverage": {"vocab_regular_rows": len(regular),
                   "vectors_found": len(covered),
                   "vectors_missing": len(regular) - len(covered),
                   "fraction": float(len(covered)) / len(regular) if regular else 1.0},
      "missing_rows": [token for token in regular if token not in covered],
      "normalization": "L2 unit norm for provided nonzero vectors; zero vectors preserved as zero",
      "zero_norm_input_vectors": zero_norm_rows,
      "special_rows": {"PAD": {"id": ENCODER_CONFIG.pad_id, "handling": "deterministic uniform[-0.05,0.05] initialization; upstream merger initializes then leaves special rows untouched"},
                       "UNK": {"id": ENCODER_CONFIG.unk_id, "handling": "deterministic uniform[-0.05,0.05] initialization; upstream merger initializes then leaves special rows untouched"}},
      "missing_vector_handling": "deterministic uniform[-0.05,0.05] initialization; seed is an explicit reproducibility choice, while upstream merger was unseeded",
      "random_seed": int(seed),
      "mapping_path": str(mapping_path),
      "mapping_sha256": sha256(mapping_path),
      "matrix_path": str(out_path),
      "matrix_shape": list(matrix.shape),
      "matrix_dtype": str(matrix.dtype),
      "matrix_sha256": sha256(out_path),
      "database_path": str(db_path),
      "database_artifact_sha256": next((x["sha256"] for x in json.load(open(db_path.parent / "artifact_manifest.json", encoding="utf-8")).get("artifacts", []) if x["path"] == "stage01.sqlite"), None),
      "raw_data_manifest_sha256": sha256(data_manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Extend the stage artifact manifest without rewriting any database rows.
    artifact_path = db_path.parent / "artifact_manifest.json"
    if artifact_path.is_file():
        artifact = json.load(open(artifact_path, encoding="utf-8"))
        entries = {x["path"]: x for x in artifact.get("artifacts", [])}
        for path in (out_path, manifest_path):
            rel = str(path.relative_to(db_path.parent))
            entries[rel] = {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
        artifact["artifacts"] = [entries[k] for k in sorted(entries)]
        artifact["entity_pretrained_available"] = True
        artifact["entity_matrix_manifest"] = str(manifest_path.relative_to(db_path.parent))
        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="artifacts/mind_large_prepared/stage01.sqlite")
    ap.add_argument("--out", default="artifacts/mind_large_prepared/embeddings/entity.npy")
    ap.add_argument("--manifest", default="artifacts/mind_large_prepared/embeddings/entity_matrix_manifest.json")
    ap.add_argument("--mapping", default="artifacts/mind_large_prepared/vocab/entity.tsv")
    ap.add_argument("--seed", type=int, default=ARTIFACT_CONFIG.initialization_seed)
    args = ap.parse_args()
    print(json.dumps(build(args.db, args.out, args.manifest, args.mapping, args.seed), indent=2))


if __name__ == "__main__":
    main()
