"""Apply the upstream per-vector L2 normalization to trained GloVe rows only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.preprocessing import normalize

from .word_embedding import read_word_vocab, sha256_file
from mind_config import ENCODER_CONFIG


SOURCE_STATUS = "MIND_TITLE_ABSTRACT_GLOVE_REIMPLEMENTATION_NOT_ORIGINAL_AUTHOR_ARTIFACT"
NORMALIZED_STATUS = "MIND_TITLE_ABSTRACT_GLOVE_REIMPLEMENTATION_TRAINED_ROWS_L2_NORMALIZED"


def normalize_artifact(source_matrix, source_manifest, vector_text, word_vocab,
                       out_matrix, out_manifest):
    source_matrix = Path(source_matrix)
    source_manifest = Path(source_manifest)
    vector_text = Path(vector_text)
    word_vocab = Path(word_vocab)
    out_matrix = Path(out_matrix)
    out_manifest = Path(out_manifest)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != SOURCE_STATUS:
        raise ValueError("source matrix must be a sidecar-verified metadata GloVe reimplementation")
    if sha256_file(str(source_matrix)) != manifest.get("matrix_sha256"):
        raise ValueError("source matrix checksum does not match its manifest")
    if sha256_file(str(word_vocab)) != manifest.get("mapping_sha256"):
        raise ValueError("current word mapping checksum differs from source matrix sidecar")
    if sha256_file(str(vector_text)) != manifest.get("trained_vectors_text_sha256"):
        raise ValueError("trained vector text differs from the source matrix sidecar")
    words, token_to_id = read_word_vocab(str(word_vocab))
    original = np.load(str(source_matrix), mmap_mode="r")
    width = int(ENCODER_CONFIG.embedding_dim)
    pad_id = int(ENCODER_CONFIG.pad_id)
    if original.shape != (len(words), width) or not np.isfinite(original).all():
        raise ValueError("source matrix shape or finite-value validation failed")
    matrix = np.array(original, dtype=np.float32, copy=True)
    ids = []
    seen = set()
    with open(vector_text, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.rstrip("\n").split()
            if len(fields) != 101:
                raise ValueError("invalid trained vector row at {}:{}".format(vector_text, line_no))
            token = fields[0]
            idx = token_to_id.get(token)
            if idx is None:
                continue
            if idx in seen:
                raise ValueError("duplicate trained vector token {}".format(token))
            seen.add(idx)
            ids.append(idx)
    if len(ids) != manifest["coverage"]["trained_vectors_found"]:
        raise ValueError("trained row count differs from the original alignment manifest")
    trained = np.asarray(ids, dtype=np.int64)
    if np.any(trained == pad_id):
        raise ValueError("PAD must never be normalized as a trained word row")
    matrix[trained] = normalize(matrix[trained], norm="l2", axis=1, copy=True)
    matrix[pad_id] = 0.0
    fallback_mask = np.ones(len(words), dtype=bool)
    fallback_mask[trained] = False
    fallback_mask[pad_id] = False
    unchanged = np.array_equal(matrix[fallback_mask], np.asarray(original[fallback_mask]))
    if not unchanged:
        raise AssertionError("fallback rows changed during trained-row normalization")
    if not np.array_equal(matrix[pad_id], np.zeros((width,), dtype=np.float32)):
        raise AssertionError("PAD row is not zero")
    if not np.isfinite(matrix).all():
        raise AssertionError("normalized matrix contains non-finite values")
    norms = np.linalg.norm(matrix[trained], axis=1)
    if len(norms) == 0 or not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise AssertionError("trained vector L2 norms are not one")
    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_matrix), matrix)
    ids_hash = hashlib.sha256(trained.astype("<u4").tobytes()).hexdigest()
    result = dict(manifest)
    result.update({
        "status": NORMALIZED_STATUS,
        "normalization": "L2 per vector only for rows present in official GloVe vectors.txt, following upstream merge-word-emb.py sklearn.preprocessing.normalize(vals.reshape(1,-1)); no matrix-wide normalization",
        "normalization_source": "upstream-mind/projects/ai/mind/src/tools/merge-word-emb.py, lines 42-47; final matrix-wide normalize is commented out at line 51",
        "normalization_runtime": {"library": "scikit-learn", "version": sklearn.__version__, "function": "sklearn.preprocessing.normalize(norm='l2', axis=1)"},
        "normalized_trained_rows": len(trained),
        "normalized_row_ids_sha256_little_endian_uint32": ids_hash,
        "trained_row_norms": {"min": float(norms.min()), "max": float(norms.max()), "mean": float(norms.mean()),
                              "max_abs_error_from_one": float(np.max(np.abs(norms - 1.0)))},
        "fallback_rows": {"count_excluding_PAD": int(fallback_mask.sum()),
                           "unchanged_byte_values_from_source_matrix": bool(unchanged),
                           "normalization_applied": False},
        "PAD": {"id": pad_id, "token": words[pad_id], "handling": "exact zero; excluded from normalization"},
        "original_matrix_path": str(source_matrix),
        "original_matrix_sha256": manifest["matrix_sha256"],
        "original_matrix_manifest_path": str(source_manifest),
        "original_matrix_manifest_sha256": sha256_file(str(source_manifest)),
        "matrix_path": str(out_matrix),
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "matrix_sha256": sha256_file(str(out_matrix)),
        "trained_vectors_text_sha256": sha256_file(str(vector_text)),
    })
    out_manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-matrix", required=True)
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--word-vocab", required=True)
    ap.add_argument("--out-matrix", required=True)
    ap.add_argument("--out-manifest", required=True)
    a = ap.parse_args()
    result = normalize_artifact(a.source_matrix, a.source_manifest, a.vectors,
                                a.word_vocab, a.out_matrix, a.out_manifest)
    summary = {k: result[k] for k in ("status", "matrix_shape", "matrix_dtype", "matrix_sha256",
                                      "original_matrix_sha256", "mapping_sha256", "normalized_trained_rows",
                                      "trained_row_norms", "fallback_rows", "PAD")}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
