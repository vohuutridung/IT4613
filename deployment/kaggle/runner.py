#!/usr/bin/env python3
"""Kaggle orchestration for preparation, training, submission, and HF upload."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANONICAL_PREPARED = ROOT / "artifacts" / "mind_large_prepared"
TOKENIZER_MANIFEST = ROOT / "tokenizer_artifact.json"
GLOVE_REPOSITORY = "https://github.com/stanfordnlp/GloVe.git"
GLOVE_COMMIT = "52f7dbc512be3f72e9c11e7cbb19dfdfa4651e9c"
DEFAULT_HF_REPO = "vohuutridung/IT4613-HGY"
PARTITIONS = range(5)
SUBMIT_MODES = ("submit", "ensemble-submit")
DEFAULT_PREDICT_BATCH_SIZE = 128
SPLITS = ("train", "dev", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class EventLog:
    def __init__(self, work_root: Path, dry_run: bool = False):
        self.work_root = work_root
        self.dry_run = bool(dry_run)
        self.events_path = work_root / "pipeline-events.jsonl"
        if not self.dry_run:
            work_root.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields) -> None:
        payload = {"event": event, "time_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
        print("[{}] {}".format(event, json.dumps(fields, ensure_ascii=False, sort_keys=True)),
              flush=True)
        if not self.dry_run:
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @contextlib.contextmanager
    def stage(self, name: str, **fields):
        started = time.monotonic()
        self.emit("START", stage=name, **fields)
        try:
            yield
        except Exception as exc:
            self.emit("FAILED", stage=name, elapsed_seconds=round(
                time.monotonic() - started, 3), error=repr(exc))
            raise
        self.emit("DONE", stage=name, elapsed_seconds=round(
            time.monotonic() - started, 3))


class Runner:
    def __init__(self, args):
        self.args = args
        self.work_root = Path(args.work_root).expanduser().resolve()
        self.log = EventLog(self.work_root, args.dry_run)
        self.python = Path(sys.executable)
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(ROOT)
        self.env["PYTHONUNBUFFERED"] = "1"

    def command(self, command, stage=None, cwd=ROOT):
        command = [str(item) for item in command]
        if self.args.dry_run:
            self.log.emit("DRY_RUN", stage=stage, command=command, cwd=str(cwd))
            return
        with self.log.stage(stage or Path(command[0]).name, command=command):
            subprocess.run(command, cwd=str(cwd), env=self.env, check=True)

    def module(self, name, *arguments, stage=None):
        self.command([self.python, "-m", name, *arguments], stage=stage)

    def _validate_raw(self, data_root: Path):
        missing = []
        for split in SPLITS:
            base = data_root / ("MINDlarge_" + split)
            for name in ("news.tsv", "behaviors.tsv", "entity_embedding.vec"):
                if not (base / name).is_file():
                    missing.append(str(base / name))
        if missing:
            raise FileNotFoundError("raw MIND layout is incomplete: {}".format(missing))

    def _link_prepared(self, target: Path, create=False):
        target = target.expanduser().resolve()
        if self.args.dry_run:
            return target
        if create:
            target.mkdir(parents=True, exist_ok=True)
        elif not target.is_dir():
            raise FileNotFoundError("prepared root does not exist: {}".format(target))
        CANONICAL_PREPARED.parent.mkdir(parents=True, exist_ok=True)
        if CANONICAL_PREPARED.is_symlink():
            if CANONICAL_PREPARED.resolve() == target:
                return target
            CANONICAL_PREPARED.unlink()
        elif CANONICAL_PREPARED.exists():
            if CANONICAL_PREPARED.resolve() == target:
                return target
            raise FileExistsError(
                "canonical prepared path already exists and is not the requested bundle: {}".format(
                    CANONICAL_PREPARED))
        CANONICAL_PREPARED.symlink_to(target, target_is_directory=True)
        return target

    def _tokenizer(self):
        from huggingface_hub import snapshot_download
        retained = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
        destination = ROOT / "artifacts/tokenizers" / (
            "bert-base-cased-" + retained["revision"])
        if not destination.is_dir():
            snapshot_download(
                repo_id="google-bert/bert-base-cased",
                revision=retained["revision"],
                local_dir=destination,
                allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
            )
        for row in retained["downloaded_files"]:
            path = ROOT / row["path"]
            if (not path.is_file() or path.stat().st_size != int(row["bytes"])
                    or sha256_file(path) != row["sha256"]):
                raise ValueError("tokenizer file differs from retained provenance: {}".format(path))

    def _glove_source(self):
        source = self.work_root / "tools" / "GloVe"
        if not source.exists():
            self.command(["git", "clone", GLOVE_REPOSITORY, source], stage="clone-glove")
            self.command(["git", "checkout", GLOVE_COMMIT], cwd=source,
                         stage="checkout-glove")
        elif not self.args.dry_run:
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
            if actual != GLOVE_COMMIT:
                raise ValueError("existing GloVe checkout is not the locked commit")
        self.command(["make", "-C", source], stage="build-glove")
        return source

    def _finalize_bundle(self, prepared: Path):
        artifact_path = prepared / "artifact_manifest.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        entries = {}
        for path in sorted(prepared.rglob("*")):
            if not path.is_file() or path.name in {
                    "artifact_manifest.json", "prepared_bundle_manifest.json"}:
                continue
            rel = path.relative_to(prepared).as_posix()
            entries[rel] = {"path": rel, "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)}
        artifact["artifacts"] = [entries[key] for key in sorted(entries)]
        artifact["word_pretrained_available"] = True
        artifact["entity_pretrained_available"] = True
        artifact_tmp = artifact_path.with_suffix(".json.tmp")
        artifact_tmp.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(artifact_tmp, artifact_path)

        from mind_config import CONFIG_DATA
        files = []
        for path in sorted(prepared.rglob("*")):
            if path.is_file() and path.name != "prepared_bundle_manifest.json":
                files.append({
                    "path": path.relative_to(prepared).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        contract = {
            "schema": "mind_compact_memmap_v1",
            "source_commit": CONFIG_DATA["source"]["commit"],
            "preprocessing": CONFIG_DATA["preprocessing"],
            "shared_architecture": {
                key: CONFIG_DATA["shared_architecture"][key]
                for key in (
                    "max_title_tokens", "max_abstract_tokens", "max_history",
                    "max_history_titles", "max_history_abstracts",
                    "max_title_entities", "max_abstract_entities",
                    "max_history_title_entities_per_news",
                    "max_history_abstract_entities_per_news", "pad_id", "unk_id")
            },
        }
        bundle = {
            "status": "MIND_15FIELD_PREPARED_BUNDLE_READY",
            "schema_version": 1,
            "preprocessing_contract": contract,
            "files": files,
        }
        payload = {row["path"]: row["sha256"] for row in files}
        bundle["payload_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        (prepared / "prepared_bundle_manifest.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return bundle

    def _verify_compact_stage(self, prepared: Path):
        manifest_path = prepared / "compact/compact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "MIND_COMPACT_MEMMAP_READY":
            raise ValueError("compact stage status is incompatible")
        observed = {}
        for row in manifest.get("files", []):
            path = prepared / "compact" / row["path"]
            if (not path.is_file() or path.stat().st_size != int(row["bytes"])
                    or sha256_file(path) != row["sha256"]):
                raise ValueError("compact stage file failed verification: {}".format(path))
            observed[row["path"]] = row["sha256"]
        payload = hashlib.sha256(json.dumps(
            observed, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if payload != manifest.get("identity", {}).get("sha256"):
            raise ValueError("compact stage payload checksum differs")
        return manifest

    def _remove_glove_intermediates(self, prepared: Path):
        """Keep provenance manifests but drop rebuild-only corpus/cooccurrence files."""
        provenance = prepared / "provenance"
        keep = {
            "title_abstract_corpus.txt.manifest.json",
            "glove/training_manifest.json",
        }
        for path in sorted(provenance.rglob("*"), reverse=True):
            if path.is_file() and path.relative_to(provenance).as_posix() not in keep:
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    def verify_bundle(self, prepared: Path):
        compact_manifest = prepared / "compact/compact_manifest.json"
        if not compact_manifest.is_file():
            raise ValueError(
                "prepared bundle does not contain the compact memmap backend")
        manifest_path = prepared / "prepared_bundle_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("prepared bundle manifest is missing: {}".format(manifest_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "MIND_15FIELD_PREPARED_BUNDLE_READY":
            raise ValueError("prepared bundle status is incompatible")
        from mind_config import CONFIG_DATA
        contract = manifest.get("preprocessing_contract", {})
        expected_shared = {
            key: CONFIG_DATA["shared_architecture"][key]
            for key in (
                "max_title_tokens", "max_abstract_tokens", "max_history",
                "max_history_titles", "max_history_abstracts",
                "max_title_entities", "max_abstract_entities",
                "max_history_title_entities_per_news",
                "max_history_abstract_entities_per_news", "pad_id", "unk_id")
        }
        if (contract.get("schema") != "mind_compact_memmap_v1"
                or contract.get("source_commit") != CONFIG_DATA["source"]["commit"]
                or contract.get("preprocessing") != CONFIG_DATA["preprocessing"]
                or contract.get("shared_architecture") != expected_shared):
            raise ValueError("prepared bundle preprocessing/source contract differs")
        hashes = {}
        for row in manifest.get("files", []):
            rel = Path(row["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("unsafe prepared bundle path: {}".format(rel))
            path = prepared / rel
            if (not path.is_file() or path.stat().st_size != int(row["bytes"])
                    or sha256_file(path) != row["sha256"]):
                raise ValueError("prepared bundle file failed verification: {}".format(path))
            hashes[rel.as_posix()] = row["sha256"]
        payload = hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
        if payload != manifest.get("payload_sha256"):
            raise ValueError("prepared bundle payload checksum differs")
        return manifest

    def prepare(self):
        if not self.args.data_root:
            raise ValueError("--data-root is required for prepare")
        data_root = Path(self.args.data_root).expanduser().resolve()
        prepared = self._link_prepared(
            Path(self.args.prepared_root).expanduser()
            if self.args.prepared_root else self.work_root / "prepared",
            create=True)
        if self.args.dry_run:
            self.log.emit("DRY_RUN", stage="prepare", data_root=str(data_root),
                          prepared_root=str(prepared))
            return prepared
        self._validate_raw(data_root)
        if (prepared / "prepared_bundle_manifest.json").is_file():
            with self.log.stage("verify-existing-prepared"):
                self.verify_bundle(prepared)
            return prepared
        if (prepared / "stage01.sqlite").exists():
            raise ValueError(
                "legacy/partial SQLite prepare output exists at {}; use a clean "
                "--work-root for the compact pipeline".format(prepared))

        with self.log.stage("tokenizer"):
            self._tokenizer()
        self.module("scripts.audit_mind_data", "--data", data_root,
                    "--out", ROOT / "artifacts/audit", stage="audit-raw")
        prepared_rel = CANONICAL_PREPARED.relative_to(ROOT)
        if (prepared / "compact/compact_manifest.json").is_file():
            with self.log.stage("verify-existing-compact-stage"):
                self._verify_compact_stage(prepared)
        else:
            self.module(
                "mind_data_pipeline.compact_preprocess",
                "--data-root", data_root,
                "--out", prepared_rel,
                "--data-manifest", ROOT / "artifacts/audit/data_manifest.json",
                stage="preprocess-compact")

        provenance = prepared_rel / "provenance"
        embeddings = prepared_rel / "embeddings"
        corpus = provenance / "title_abstract_corpus.txt"
        glove_out = provenance / "glove"
        word_vocab = prepared_rel / "vocab/word.tsv"
        source = self._glove_source()
        self.module("mind_news_metadata.word_embedding", "corpus",
                    "--compact-root", prepared_rel,
                    "--word-vocab", word_vocab, "--out", corpus,
                    stage="build-glove-corpus")
        self.module("mind_news_metadata.word_embedding", "train",
                    "--corpus", corpus, "--out-dir", glove_out,
                    "--source-dir", source, stage="train-glove")
        self.module("mind_news_metadata.word_embedding", "align",
                    "--word-vocab", word_vocab,
                    "--vectors", glove_out / "vectors.txt",
                    "--matrix", embeddings / "word_glove_title_abstract.npy",
                    "--manifest", embeddings / "word_glove_title_abstract.manifest.json",
                    "--corpus-manifest", Path(str(corpus) + ".manifest.json"),
                    "--training-manifest", glove_out / "training_manifest.json",
                    stage="align-glove")
        self.module("mind_news_metadata.normalize_word_matrix",
                    "--source-matrix", embeddings / "word_glove_title_abstract.npy",
                    "--source-manifest", embeddings / "word_glove_title_abstract.manifest.json",
                    "--vectors", glove_out / "vectors.txt",
                    "--word-vocab", word_vocab,
                    "--out-matrix", embeddings / "word_glove_title_abstract_l2.npy",
                    "--out-manifest", embeddings / "word_glove_title_abstract_l2.manifest.json",
                    stage="normalize-glove")
        self.module("mind_news_metadata.build_entity_matrix",
                    "--compact-root", prepared_rel,
                    "--out", embeddings / "entity.npy",
                    "--manifest", embeddings / "entity_matrix_manifest.json",
                    "--mapping", prepared_rel / "vocab/entity.tsv",
                    "--data-root", data_root, stage="build-entity-matrix")
        with self.log.stage("finalize-prepared-bundle"):
            self._remove_glove_intermediates(prepared)
            self._finalize_bundle(prepared)
            self.verify_bundle(prepared)
        return prepared

    def use_prepared(self):
        if self.args.prepared_root:
            prepared = self._link_prepared(Path(self.args.prepared_root), create=False)
        elif CANONICAL_PREPARED.is_dir():
            prepared = CANONICAL_PREPARED.resolve()
        else:
            raise FileNotFoundError("no prepared bundle; pass --prepared-root or run prepare")
        if not self.args.dry_run:
            with self.log.stage("verify-prepared-bundle"):
                self.verify_bundle(prepared)
        return prepared

    def _copy_resume(self, resume_source: Path, output: Path):
        if output.exists():
            raise FileExistsError("resume output already exists: {}".format(output))
        shutil.copytree(resume_source, output)

    def _gpu_preflight(self, required=1):
        if self.args.allow_cpu:
            self.log.emit("SKIP", stage="gpu-preflight", reason="--allow-cpu")
            return
        probe = (
            "import json,tensorflow as tf; "
            "g=tf.config.list_physical_devices('GPU'); "
            "assert len(g)>=" + str(required) + ", 'Not enough visible GPUs: '+str(g); "
            "[tf.config.experimental.set_memory_growth(x,True) for x in g]; "
            "print(json.dumps({'tensorflow':tf.__version__,'gpus':[x.name for x in g]}))"
        )
        self.command([self.python, "-c", probe], stage="gpu-preflight")

    def train(self, partition: int):
        self.use_prepared()
        self._gpu_preflight(required=self.args.replicas)
        run_dir = self.work_root / "recommendation_runs" / ("partition-{}".format(partition))
        progress_path = run_dir / "training_progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("full_epoch"):
                self.log.emit("SKIP", stage="train", partition=partition,
                              reason="full epoch already complete")
                return run_dir
        resume = False
        if self.args.resume_run:
            source = Path(self.args.resume_run).expanduser().resolve()
            if source.name.startswith("partition-") and source.name != run_dir.name:
                candidate = source.parent / run_dir.name
                if not candidate.is_dir():
                    raise ValueError(
                        "resume source {} is for another partition and {} is absent".format(
                            source, candidate))
                source = candidate
            elif source.name == "recommendation_runs" or not source.name.startswith("partition-"):
                if not (source / run_dir.name).is_dir():
                    raise FileNotFoundError(
                        "resume root has no {}".format(run_dir.name))
                source = source / run_dir.name
            if not source.is_dir() and not self.args.dry_run:
                raise FileNotFoundError("resume run does not exist: {}".format(source))
            if run_dir.exists() and source.resolve() != run_dir.resolve():
                raise FileExistsError(
                    "working run already exists; refusing to ignore/merge resume source: {}".format(
                        run_dir))
            if not run_dir.exists() and not self.args.dry_run:
                self._copy_resume(source, run_dir)
            resume = True
        elif run_dir.is_dir():
            if not (run_dir / "checkpoints/checkpoint").is_file():
                raise ValueError("existing partial run has no checkpoint manager state")
            resume = True
        command = [
            self.python, "-m", "mind_news_metadata.full_pipeline", "train",
            "--db", CANONICAL_PREPARED,
            "--partition", partition,
            "--seed", self.args.seed,
            "--effective-batch", self.args.effective_batch,
            "--batch-size", self.args.microbatch,
            "--replicas", self.args.replicas,
            "--precision", self.args.precision,
            "--prefetch-batches", self.args.prefetch_batches,
            "--checkpoint-every", self.args.checkpoint_every,
            "--train-step-xla" if self.args.train_step_xla else "--no-train-step-xla",
            "--output-dir", run_dir,
        ]
        if self.args.allow_cpu:
            command.append("--allow-cpu")
        if resume:
            command.append("--resume")
        if self.args.max_updates is not None:
            command.extend(["--max-updates", self.args.max_updates])
        if self.args.max_impressions is not None:
            command.extend(["--max-impressions", self.args.max_impressions])
        if self.args.stop_after_updates_this_run is not None:
            command.extend(["--stop-after-updates-this-run", self.args.stop_after_updates_this_run])
        self.command(command, stage="train-partition-{}".format(partition))
        return run_dir

    def _prediction(self, run_dir: Path, index: int):
        progress_path = run_dir / "training_progress.json"
        if not (self.args.dry_run and self.args.hf_folder):
            if not progress_path.is_file():
                raise FileNotFoundError("run has no training_progress.json: {}".format(run_dir))
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if (not progress.get("full_epoch") and not self.args.allow_partial
                    and self.args.predict_max_candidates is None):
                raise ValueError(
                    "refusing test submission from partial run {}; use --allow-partial explicitly".format(
                        run_dir))
        membership = CANONICAL_PREPARED / "membership/train_did_membership.npy"
        suffix = "test-smoke" if self.args.predict_max_candidates is not None else "test"
        output = self.work_root / "predictions" / (
            "model-{}-{}.csv.gz".format(index, suffix))
        limit_args = []
        if self.args.predict_max_candidates is not None:
            limit_args = ["--max-candidates", self.args.predict_max_candidates]
        self.module("mind_news_metadata.full_pipeline", "predict",
                    "--run-dir", run_dir,
                    "--db", CANONICAL_PREPARED,
                    "--split", "test", "--profile", "test",
                    "--batch-size", self.args.predict_batch_size,
                    "--replicas", self.args.predict_replicas,
                    "--train-dids", membership, "--output", output,
                    *limit_args,
                    stage="predict-test-model-{}".format(index))
        return output

    def _download_models(self, ensemble=False):
        folder = PurePosixPath(self.args.hf_folder)
        if (folder.is_absolute() or ".." in folder.parts or str(folder) == "."
                or any(char in str(folder) for char in "*?[]")):
            raise ValueError("--hf-folder must be a safe relative repository path")
        folders = [folder / "partition-{}".format(i) for i in PARTITIONS] if ensemble else [folder]
        cache_dir = self.work_root / "hf-cache"
        source = {"repo": self.args.hf_repo, "folder": str(folder),
                  "revision": self.args.hf_revision}
        if self.args.dry_run:
            for run_folder in folders:
                self.log.emit("DRY_RUN", stage="download-hf-model",
                              **{**source, "folder": str(run_folder)})
            return [cache_dir / "<snapshot>" / run_folder for run_folder in folders]

        from huggingface_hub import HfApi, hf_hub_download, snapshot_download

        token = os.environ.get("HF_TOKEN")
        with self.log.stage("download-hf-model", **source):
            # Resolve the branch or tag once for every partition and file.
            revision = HfApi(token=token).model_info(
                self.args.hf_repo, revision=self.args.hf_revision).sha
            if not revision:
                raise ValueError("could not resolve the Hugging Face model revision")
            download = {"repo_id": self.args.hf_repo, "repo_type": "model",
                        "revision": revision, "cache_dir": cache_dir, "token": token}
            metadata = ["run_config.json", "training_progress.json"]
            prefixes = {}
            patterns = []
            for run_folder in folders:
                checkpoint_dir = run_folder / "checkpoints"
                state_name = (checkpoint_dir / "checkpoint").as_posix()
                state = Path(hf_hub_download(filename=state_name, **download))
                match = re.search(r'^model_checkpoint_path: "(ckpt-\d+)"$',
                                  state.read_text(encoding="utf-8"), re.MULTILINE)
                if not match:
                    raise ValueError(
                        "{} must point to a relative ckpt-N".format(state_name))
                prefix = checkpoint_dir / match[1]
                prefixes[run_folder] = prefix
                patterns.extend(str(run_folder / name) for name in metadata)
                patterns.extend([str(prefix) + ".index", str(prefix) + ".data-*"])
            snapshot = Path(snapshot_download(allow_patterns=patterns, **download))
            run_dirs = []
            for run_folder, prefix in prefixes.items():
                run_dir = snapshot / run_folder
                missing = [name for name in metadata if not (run_dir / name).is_file()]
                if missing:
                    raise FileNotFoundError(
                        "Hugging Face run {} is missing: {}".format(run_folder, ", ".join(missing)))
                checkpoint = snapshot / prefix
                shards = sorted(path.name for path in
                                checkpoint.parent.glob(checkpoint.name + ".data-*") if path.is_file())
                expected = ["{}.data-{:05d}-of-{:05d}".format(checkpoint.name, i, len(shards))
                            for i in range(len(shards))]
                if (not (checkpoint.parent / "checkpoint").is_file()
                        or not checkpoint.with_suffix(".index").is_file()
                        or not shards or shards != expected):
                    raise FileNotFoundError("Hugging Face run is missing checkpoint files: {}".format(prefix))
                self.log.emit("HF_MODEL_READY", repo=self.args.hf_repo, folder=str(run_folder),
                              revision=revision, checkpoint=str(prefix))
                run_dirs.append(run_dir)
            return run_dirs

    def submit(self, run_dirs, ensemble=False):
        self.use_prepared()
        if self.args.hf_folder:
            run_dirs = self._download_models(ensemble=ensemble)
        if not run_dirs:
            if ensemble:
                run_dirs = [self.work_root / "recommendation_runs" / ("partition-{}".format(i))
                            for i in PARTITIONS]
            else:
                run_dirs = [self.work_root / "recommendation_runs" / (
                    "partition-{}".format(self.args.partition))]
        predictions = [self._prediction(Path(run_dir), index)
                       for index, run_dir in enumerate(run_dirs)]
        if self.args.predict_max_candidates is not None:
            self.log.emit("SMOKE_PREDICTION_COMPLETE", candidate_limit=self.args.predict_max_candidates,
                          predictions=[str(path) for path in predictions], full_split=False,
                          submission_created=False)
            return predictions
        output = Path(self.args.output).expanduser() if self.args.output else (
            self.work_root / "submissions" / (
                "ensemble-submission.zip" if ensemble else
                "partition-{}-submission.zip".format(self.args.partition)))
        self.module("mind_news_metadata.full_pipeline", "submit",
                    "--inputs", *predictions,
                    "--db", CANONICAL_PREPARED,
                    "--output", output, stage="create-codalab-submission")
        return output

    def push_model(self):
        if not self.args.run_dir:
            raise ValueError("push-model requires --run-dir")
        if not self.args.hf_folder:
            raise ValueError("push-model requires --hf-folder")
        folder = PurePosixPath(self.args.hf_folder)
        if folder.is_absolute() or ".." in folder.parts or str(folder) in ("", "."):
            raise ValueError("--hf-folder must be a safe relative repository path")
        token = os.environ.get("HF_TOKEN")
        if not token and not self.args.dry_run:
            raise ValueError("HF_TOKEN is missing; configure it as a Kaggle Secret")
        run_dir = Path(self.args.run_dir[0]).expanduser().resolve()
        progress_path = run_dir / "training_progress.json"
        if not progress_path.is_file():
            raise FileNotFoundError("training_progress.json is missing")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not progress.get("full_epoch") and not self.args.allow_partial:
            raise ValueError("refusing to upload a partial run without --allow-partial")
        required = ["run_config.json", "partition_counts.json", "training_progress.json"]
        files = [run_dir / name for name in required]
        files += sorted((run_dir / "checkpoints").glob("*"))
        files += sorted(run_dir.glob("*.log"))
        if any(not path.is_file() for path in files[:len(required)]) or not any(
                path.suffix == ".index" for path in files):
            raise ValueError("run metadata or checkpoint shards are incomplete")
        if self.args.dry_run:
            self.log.emit("DRY_RUN", stage="push-model", repo=self.args.hf_repo,
                          folder=str(folder), files=[str(path) for path in files])
            return
        from huggingface_hub import HfApi
        with tempfile.TemporaryDirectory(dir=self.work_root) as temporary:
            staging = Path(temporary)
            hashes = []
            for source in files:
                relative = source.relative_to(run_dir)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                hashes.append({"path": relative.as_posix(), "bytes": source.stat().st_size,
                               "sha256": sha256_file(source)})
            api = HfApi(token=token)
            api.create_repo(self.args.hf_repo, repo_type="model", exist_ok=True)
            result = api.upload_folder(
                repo_id=self.args.hf_repo, repo_type="model",
                folder_path=staging, path_in_repo=str(folder),
                commit_message="Upload MIND run {}".format(run_dir.name))
        receipt = {
            "status": "HF_MODEL_RUN_UPLOADED",
            "repo_id": self.args.hf_repo,
            "folder": str(folder),
            "commit_url": getattr(result, "commit_url", None),
            "commit_oid": getattr(result, "oid", None),
            "files": hashes,
            "uploaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (run_dir / "hf_upload_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.log.emit("UPLOADED", repo=self.args.hf_repo, folder=str(folder),
                      commit_url=receipt["commit_url"])

    def execute(self):
        mode = self.args.mode
        if mode == "prepare":
            self.prepare()
        elif mode == "train":
            self.train(self.args.partition)
        elif mode == "benchmark":
            self.use_prepared()
            self.command([
                self.python, ROOT / "scripts/benchmark_training.py",
                "--prepared-root", CANONICAL_PREPARED,
                "--work-root", self.work_root / "benchmarks",
                "--effective-batch", self.args.effective_batch,
                "--batch-size", self.args.microbatch, "--replicas", self.args.replicas,
                "--updates", self.args.benchmark_updates, "--partition", self.args.partition,
                "--seed", self.args.seed, "--prefetch-batches", self.args.prefetch_batches,
                "--checkpoint-every", self.args.checkpoint_every], stage="benchmark-training")
        elif mode == "train-ensemble":
            for partition in PARTITIONS:
                self.train(partition)
        elif mode == "submit":
            self.submit([Path(path) for path in self.args.run_dir], ensemble=False)
        elif mode == "ensemble-submit":
            self.submit([Path(path) for path in self.args.run_dir], ensemble=True)
        elif mode == "single":
            if self.args.prepared_root:
                self.use_prepared()
            else:
                self.prepare()
            run_dir = self.train(self.args.partition)
            self.submit([run_dir], ensemble=False)
        elif mode == "push-model":
            self.push_model()
        else:
            raise AssertionError(mode)


def parser():
    default_work = Path("/kaggle/working/mind15") if Path("/kaggle/working").is_dir() else (
        ROOT / "artifacts/kaggle_work")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=(
        "prepare", "train", "train-ensemble", "submit",
        "ensemble-submit", "single", "push-model", "benchmark"))
    ap.add_argument("--data-root")
    ap.add_argument("--prepared-root")
    ap.add_argument("--work-root", default=str(default_work))
    ap.add_argument("--partition", type=int, default=0, choices=PARTITIONS)
    from mind_config import EXECUTION_CONFIG, TRAINING_CONFIG
    ap.add_argument("--effective-batch", type=int, default=TRAINING_CONFIG["effective_batch_size"])
    ap.add_argument("--batch-size", "--microbatch", dest="microbatch", type=int,
                    default=TRAINING_CONFIG["microbatch_size"], help="microbatch size per GPU")
    ap.add_argument("--predict-batch-size", type=int, default=DEFAULT_PREDICT_BATCH_SIZE,
                    help="global test inference batch size across prediction GPUs (default: 128)")
    ap.add_argument("--predict-replicas", type=int, default=1,
                    help="number of GPUs running the same checkpoint in parallel (default: 1)")
    ap.add_argument("--predict-max-candidates", type=int,
                    help="smoke predict at most N candidates per model and finish before ZIP creation")
    ap.add_argument("--replicas", type=int, default=TRAINING_CONFIG["replicas"])
    ap.add_argument("--precision", choices=("float32", "mixed_float16"), default=TRAINING_CONFIG["precision"])
    ap.add_argument("--prefetch-batches", type=int, default=EXECUTION_CONFIG.prefetch_batches)
    ap.add_argument("--checkpoint-every", type=int, default=EXECUTION_CONFIG.checkpoint_every_updates)
    ap.add_argument("--train-step-xla", action=argparse.BooleanOptionalAction,
                    default=TRAINING_CONFIG["train_step_xla"])
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--max-updates", type=int)
    ap.add_argument("--max-impressions", type=int,
                    help="limit train impressions for smoke tests, including partition counting")
    ap.add_argument("--stop-after-updates-this-run", type=int)
    ap.add_argument("--benchmark-updates", type=int, default=30)
    ap.add_argument("--resume-run")
    ap.add_argument("--run-dir", action="append", default=[])
    ap.add_argument("--output")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--hf-folder",
                    help="Hugging Face run folder, or parent of partition-0..4 for ensemble-submit")
    ap.add_argument("--hf-revision", default="main",
                    help="Hugging Face branch, tag or commit to predict from (default: main)")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if args.hf_folder is not None and args.mode not in (*SUBMIT_MODES, "push-model"):
        ap.error("--hf-folder is only supported with --mode submit, ensemble-submit or push-model")
    if args.mode in SUBMIT_MODES and args.hf_folder is not None:
        if not args.hf_folder.strip():
            ap.error("--hf-folder cannot be empty")
        if args.run_dir:
            ap.error("choose either --hf-folder or --run-dir for --mode {}".format(args.mode))
    if args.hf_revision != "main" and not (args.mode in SUBMIT_MODES and args.hf_folder):
        ap.error("--hf-revision requires --mode submit or ensemble-submit with --hf-folder")
    if args.predict_batch_size < 1:
        ap.error("--predict-batch-size must be positive")
    if args.predict_replicas < 1:
        ap.error("--predict-replicas must be positive")
    if args.predict_max_candidates is not None and args.predict_max_candidates < 1:
        ap.error("--predict-max-candidates must be positive")
    if args.max_impressions is not None and args.max_impressions < 1:
        ap.error("--max-impressions must be positive")
    if args.mode in ("train", "train-ensemble", "single", "benchmark"):
        from mind_news_metadata.training_runtime import batch_layout
        batch_layout(args.effective_batch, args.microbatch, args.replicas)
        if args.prefetch_batches < 0 or args.checkpoint_every < 1:
            raise ValueError("invalid prefetch depth or checkpoint interval")
    Runner(args).execute()


if __name__ == "__main__":
    main()
