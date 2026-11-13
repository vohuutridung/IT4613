"""Source-guided GloVe preparation for MIND title/abstract metadata.

This trains word vectors only. It does not use clicks or recommendation losses.
The corpus contains two newline-delimited documents per unique news row.
"""
import argparse
import collections
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from mind_config import ARTIFACT_CONFIG, ENCODER_CONFIG, PREPROCESSING_CONFIG


SEED = ARTIFACT_CONFIG.initialization_seed
WIDTH = ENCODER_CONFIG.embedding_dim
WINDOW = 15
MIN_COUNT = 1
ITERATIONS = 15
X_MAX = 10.0
ALPHA = 0.75
THREADS = 4
MEMORY_GB = 1.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_word_vocab(path):
    by_id, by_token = [], {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError("invalid word vocabulary row {}:{}".format(path, line_no))
            idx, token = int(parts[0]), parts[1]
            if idx != len(by_id) or token in by_token:
                raise ValueError("word vocabulary must have unique tokens and contiguous IDs")
            by_id.append(token)
            by_token[token] = idx
    if (len(by_id) <= ENCODER_CONFIG.pad_id
            or by_id[ENCODER_CONFIG.pad_id] != "[PAD]"):
        raise ValueError("locked tokenizer vocabulary must use [PAD] at the configured PAD ID")
    return by_id, by_token


def build_corpus(db_path, word_vocab_path, out_path, limit_news=None):
    """Stream title/abstract token IDs from unique news lookup; never reads labels."""
    words, _ = read_word_vocab(word_vocab_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db_uri = "file:{}?mode=ro".format(Path(db_path).resolve())
    conn = sqlite3.connect(db_uri, uri=True)
    split_counts = collections.Counter()
    token_counts = collections.Counter()
    news_count = 0
    title_docs = abstract_docs = 0
    token_total = 0
    started = time.time()
    query = "SELECT news_id,source_split,title_ids,abstract_ids FROM news ORDER BY did"
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for news_id, split, title_blob, abstract_blob in conn.execute(query):
            if limit_news is not None and news_count >= int(limit_news):
                break
            for field, blob in (("title", title_blob), ("abstract", abstract_blob)):
                ids = np.frombuffer(blob or b"", dtype="<u4")
                ids = ids[ids != 0]
                try:
                    tokens = [words[int(i)] for i in ids]
                except IndexError as exc:
                    raise ValueError("stored word token ID outside locked vocabulary for {}".format(news_id)) from exc
                out.write(" ".join(tokens) + "\n")
                token_counts[field] += len(tokens)
                token_total += len(tokens)
                if field == "title":
                    title_docs += 1
                else:
                    abstract_docs += 1
            split_counts[split] += 1
            news_count += 1
            if news_count % 25000 == 0:
                print("[corpus] news={} tokens={}".format(news_count, token_total), file=sys.stderr, flush=True)
    conn.close()
    manifest = {
        "status": "MIND_NEWS_METADATA_CORPUS_NO_CLICK_LABELS",
        "source_database": str(Path(db_path)),
        "source_database_sha256": sha256_file(db_path),
        "source_table": "news (unique news_id lookup)",
        "source_split_counts": dict(split_counts),
        "metadata_scope": "train+dev+test news metadata; first-occurrence rows already resolved by stage-01 policy",
        "news_rows": news_count,
        "documents": {"title": title_docs, "abstract": abstract_docs, "total": title_docs + abstract_docs},
        "tokens_nonpadding": dict(token_counts),
        "tokens_total": token_total,
        "document_boundary": "one newline per field; co-occurrence contexts do not cross fields",
        "token_source": "stored locked tokenizer token IDs in news.title_ids/news.abstract_ids; [PAD] ID 0 omitted",
        "tokenizer_vocab_path": str(Path(word_vocab_path)),
        "tokenizer_vocab_sha256": sha256_file(word_vocab_path),
        "click_or_label_data_read": False,
        "corpus_path": str(out_path),
        "corpus_bytes": out_path.stat().st_size,
        "corpus_sha256": sha256_file(str(out_path)),
        "limit_news": limit_news,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    manifest_path = str(out_path) + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def run_glove(corpus, out_dir, source_dir, iterations=ITERATIONS, threads=THREADS,
              memory_gb=MEMORY_GB, seed=SEED, smoke=False):
    """Run only Stanford's official GloVe C tools; demo.sh/text8 is never used."""
    source_dir = Path(source_dir).resolve()
    build = source_dir / "build"
    required = [build / name for name in ("vocab_count", "cooccur", "shuffle", "glove")]
    if not all(p.is_file() and os.access(str(p), os.X_OK) for p in required):
        raise ValueError("build official Stanford GloVe first: make -C {}".format(source_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = out_dir / "glove_vocab.txt"
    cooc = out_dir / "cooccurrence.bin"
    shuffled = out_dir / "cooccurrence.shuf.bin"
    vectors_base = out_dir / "vectors"
    overflow = out_dir / "overflow"
    temp_shuffle = out_dir / "temp_shuffle"
    commands = []

    with open(corpus, "rb") as inp, open(vocab, "wb") as out:
        cmd = [str(build / "vocab_count"), "-min-count", str(MIN_COUNT), "-verbose", "1"]
        subprocess.check_call(cmd, stdin=inp, stdout=out)
    commands.append([str(build / "vocab_count"), "-min-count", str(MIN_COUNT), "-verbose", "1", "<", str(corpus), ">", str(vocab)])

    with open(corpus, "rb") as inp, open(cooc, "wb") as out:
        cmd = [str(build / "cooccur"), "-memory", str(memory_gb), "-vocab-file", str(vocab),
               "-verbose", "1", "-window-size", str(WINDOW), "-overflow-file", str(overflow)]
        subprocess.check_call(cmd, stdin=inp, stdout=out)
    commands.append([str(build / "cooccur"), "-memory", str(memory_gb), "-vocab-file", str(vocab),
                     "-verbose", "1", "-window-size", str(WINDOW), "-overflow-file", str(overflow),
                     "<", str(corpus), ">", str(cooc)])

    with open(cooc, "rb") as inp, open(shuffled, "wb") as out:
        cmd = [str(build / "shuffle"), "-memory", str(memory_gb), "-seed", str(seed),
               "-temp-file", str(temp_shuffle), "-verbose", "1"]
        subprocess.check_call(cmd, stdin=inp, stdout=out)
    commands.append([str(build / "shuffle"), "-memory", str(memory_gb), "-seed", str(seed),
                     "-temp-file", str(temp_shuffle), "-verbose", "1", "<", str(cooc), ">", str(shuffled)])

    with open(os.devnull, "wb") as devnull:
        cmd = [str(build / "glove"), "-save-file", str(vectors_base), "-threads", str(threads),
               "-input-file", str(shuffled), "-x-max", str(X_MAX), "-alpha", str(ALPHA),
               "-iter", str(iterations), "-vector-size", str(WIDTH), "-binary", "0",
               "-model", "2", "-vocab-file", str(vocab), "-seed", str(seed), "-verbose", "1"]
        subprocess.check_call(cmd, stdout=devnull)
    commands.append(cmd)
    vector_text = Path(str(vectors_base) + ".txt")
    if not vector_text.is_file():
        raise RuntimeError("official GloVe did not produce expected text vectors: {}".format(vector_text))
    run_manifest = {
        "status": "SMOKE_TEST_ONLY" if smoke else "TRAINED_REIMPLEMENTATION_GLOVE",
        "official_source": "Stanford NLP GloVe C implementation",
        "official_source_directory": str(source_dir),
        "official_source_commit": subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"]).decode().strip(),
        "official_source_file_checksums": {
            str(p.relative_to(source_dir)): sha256_file(str(p))
            for p in [source_dir / "Makefile"] + sorted((source_dir / "src").glob("*"))
            if p.is_file()
        },
        "built_binary_checksums": {p.name: sha256_file(str(p)) for p in required},
        "parameters": {"vector_size": WIDTH, "window_size": WINDOW, "min_count": MIN_COUNT,
                       "iterations": int(iterations), "x_max": X_MAX, "alpha": ALPHA,
                       "threads": int(threads), "memory_gb_each_tool": float(memory_gb),
                       "seed": int(seed), "model": 2, "binary": 0,
                       "symmetric_context": True, "inverse_distance_weighting": True},
        "parameter_provenance": {
            "vector_size": "MIND encoder embedding dimension is {}; upstream glove-100.sh demo uses 50 and is not run".format(WIDTH),
            "window_size": "15 taken from upstream glove-100.sh example as a source-grounded choice",
            "x_max": "10 taken from upstream glove-100.sh example as a source-grounded choice",
            "min_count": "1 reimplementation choice to retain every observed tokenizer token",
            "iterations": "15 taken from upstream glove-100.sh example",
            "alpha": "0.75 official Stanford GloVe default",
            "memory_gb_each_tool": "1.0 resource-bound reimplementation choice",
            "threads": "4 resource-bound reimplementation choice",
            "seed": "20261024 explicit reproducibility choice; upstream demo does not fix a seed",
            "model": "2 official GloVe word+context vector sum output",
        },
        "numerical_reproducibility": "seed fixes C PRNG initial state and shuffle; multi-thread Hogwild updates may vary slightly with thread schedule",
        "commands": commands,
        "files": {name: {"bytes": p.stat().st_size, "sha256": sha256_file(str(p))}
                  for name, p in (("corpus", Path(corpus)), ("vocab", vocab), ("cooccurrence", cooc),
                                  ("shuffled", shuffled), ("vectors_text", vector_text))},
    }
    with open(out_dir / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return run_manifest


def align_vectors(word_vocab_path, vector_text_path, matrix_path, manifest_path,
                  corpus_manifest_path, training_manifest_path, fallback_seed=SEED):
    words, token_to_id = read_word_vocab(word_vocab_path)
    corpus_m = json.load(open(corpus_manifest_path, encoding="utf-8"))
    training_m = json.load(open(training_manifest_path, encoding="utf-8"))
    if training_m.get("status") != "TRAINED_REIMPLEMENTATION_GLOVE":
        raise ValueError("smoke-test GloVe output cannot be installed as a production matrix")
    if int(training_m["parameters"]["vector_size"]) != WIDTH:
        raise ValueError("GloVe vector width is not the configured {}".format(WIDTH))
    matrix = np.random.RandomState(int(fallback_seed)).uniform(-0.05, 0.05,
                    size=(len(words), WIDTH)).astype(np.float32)
    matrix[ENCODER_CONFIG.pad_id] = 0.0  # tokenizer [PAD]
    found = set()
    duplicates = 0
    with open(vector_text_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split()
            if len(parts) != WIDTH + 1:
                raise ValueError("invalid GloVe row {}:{} expected token + {} floats".format(vector_text_path, line_no, WIDTH))
            token = parts[0]
            # token IDs are exact map keys; GloVe tokens outside the current tokenizer are ignored.
            idx = token_to_id.get(token)
            if idx is None:
                continue
            if idx in found:
                duplicates += 1
                continue
            vector = np.asarray(parts[1:], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError("non-finite GloVe vector for token {}".format(token))
            matrix[idx] = vector
            found.add(idx)
    missing_ids = sorted(set(range(len(words))) - found - {ENCODER_CONFIG.pad_id})
    matrix_path = Path(matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(matrix_path), matrix)
    manifest = {
        "status": "MIND_TITLE_ABSTRACT_GLOVE_REIMPLEMENTATION_NOT_ORIGINAL_AUTHOR_ARTIFACT",
        "training_source_manifest": str(training_manifest_path),
        "training_source_manifest_sha256": sha256_file(training_manifest_path),
        "trained_vectors_text_sha256": sha256_file(vector_text_path),
        "corpus_manifest": str(corpus_manifest_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "corpus_sha256": corpus_m["corpus_sha256"],
        "metadata_scope": corpus_m["metadata_scope"],
        "click_or_label_data_used": False,
        "mapping_path": str(word_vocab_path),
        "mapping_sha256": sha256_file(word_vocab_path),
        "matrix_path": str(matrix_path),
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "matrix_sha256": sha256_file(str(matrix_path)),
        "tokenizer_revision": PREPROCESSING_CONFIG.tokenizer_revision,
        "coverage": {"vocabulary_rows_excluding_PAD": len(words) - 1,
                     "trained_vectors_found": len(found),
                     "fraction": len(found) / float(max(1, len(words) - 1))},
        "missing_vector_count_excluding_PAD": len(missing_ids),
        "missing_tokens": [words[i] for i in missing_ids],
        "duplicate_vector_rows_after_first": duplicates,
        "PAD": {"id": ENCODER_CONFIG.pad_id, "token": words[ENCODER_CONFIG.pad_id], "handling": "exact zero vector"},
        "UNK": {"id": token_to_id.get("[UNK]"), "handling": "trained if observed; otherwise deterministic uniform fallback"},
        "fallback": "RandomState(seed).uniform(-0.05,0.05), used for every tokenizer row without a trained GloVe vector",
        "fallback_seed": int(fallback_seed),
        "normalization": "none after GloVe output; model=2 exports word+context sum",
        "training_normalization": "none",
        "dtype": "float32",
        "trainable_in_encoder": True,
        "source_embeddings_learned": "GloVe co-occurrence objective over title/abstract metadata only; no recommendation objective",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command")
    c = sub.add_parser("corpus")
    c.add_argument("--db", required=True)
    c.add_argument("--word-vocab", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--limit-news", type=int)
    t = sub.add_parser("train")
    t.add_argument("--corpus", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--source-dir", required=True)
    t.add_argument("--iterations", type=int, default=ITERATIONS)
    t.add_argument("--threads", type=int, default=THREADS)
    t.add_argument("--memory-gb", type=float, default=MEMORY_GB)
    t.add_argument("--seed", type=int, default=SEED)
    t.add_argument("--smoke", action="store_true")
    m = sub.add_parser("align")
    m.add_argument("--word-vocab", required=True)
    m.add_argument("--vectors", required=True)
    m.add_argument("--matrix", required=True)
    m.add_argument("--manifest", required=True)
    m.add_argument("--corpus-manifest", required=True)
    m.add_argument("--training-manifest", required=True)
    m.add_argument("--fallback-seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.command == "corpus":
        result = build_corpus(a.db, a.word_vocab, a.out, a.limit_news)
    elif a.command == "train":
        result = run_glove(a.corpus, a.out_dir, a.source_dir, a.iterations,
                           a.threads, a.memory_gb, a.seed, a.smoke)
    elif a.command == "align":
        result = align_vectors(a.word_vocab, a.vectors, a.matrix, a.manifest,
                              a.corpus_manifest, a.training_manifest, a.fallback_seed)
    else:
        ap.error("choose corpus or train")
    if a.command == "align":
        result = {k: result[k] for k in ("status", "matrix_shape", "matrix_dtype", "matrix_sha256",
                                          "mapping_sha256", "coverage", "missing_vector_count_excluding_PAD",
                                          "fallback_seed", "PAD", "UNK")}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
