# Cleanup log

Date: 2026-11-12. Local source-tree cleanup only. No Kaggle/Hugging Face resource, raw MIND file, SQLite DB/WAL/SHM, vocabulary, mapping, tokenizer asset, pretrained matrix, upstream checkout, source lock, or training checkpoint was changed.

## Consolidated

- `configs/mind_no_body_15field.json` is the single parameter source; `mind_config.py` exposes stage-01, active encoder, training, and execution views. A static comparison found all 34 stage-01 defaults equal to the prior recorded `MindLargePreparationConfig` values.
- The retained core is `mind_data_pipeline/` plus `mind_news_metadata/`. The active variant has 15 fields and intentionally omits body. Source differences, including the unresolved `bert-adam` binding and private pooling behavior, are in `docs/source_reimplementation_matrix.md`.
- Updated the prepared-artifact manifests to stop listing removed forward/regression reports. The legacy stage-01 code lineage entries needed by its reconciliation path remain in `artifacts/project_manifest.json`.

## Removed from the working tree

- Kaggle/GPU/restore/microbatch/budget tooling: `deployment/kaggle/`, its notebooks and runner payloads, and root `notebookede77809af.ipynb`.
- Redundant or superseded status documents: the old SOPs, compatibility/dependency/blocker/validation reports, name map, old technical/project reports, publication and performance/training reports, archived 17-field review, EDA summary, and duplicate root field contract/config snapshots.
- Diagnostic-only code: `mind_news_metadata/artifact_manifest.py`, `data_preparation.py`, `forward.py`, smoke/oracle helpers; GPU, legacy dependency, tokenizer-check and EDA scripts. The primary preprocessing, artifact preparation, validation, regression, training, prediction and ensemble modules remain.
- Superseded local experiment artifacts: encoder fixtures, GloVe smoke output, tokenizer/melt/body-crawler diagnostics, microbatch/resume/parity runs, distribution pilot, stale run summary and prior forward reports. The exact retained real checkpoint is `artifacts/recommendation_runs/partition-0-cpu-full/`.
- Project-owned `__pycache__` under the active source/script directories.

The untracked user notebooks removed from the tree have scoped recovery copies in `/tmp/mind_project_cleanup_recovery_20261112/user_notebooks/`. Rewritten current documents/config/source files were copied before editing to `/tmp/mind_project_cleanup_recovery_20261112/current_versions/`. These are narrow recovery copies, not an archive of the experiment tree.

## Preserved

- Raw `data/MINDlarge_{train,dev,test}` and the local sample/data dictionary.
- `artifacts/mind_large_prepared/`: active DB and sidecars, `resolved_config.json`, `state.json`, vocabulary, mappings, tokenizer-linked metadata, GloVe/entity matrices and provenance, audit/data manifest, stage-01 validation, and legacy reconciliation evidence.
- `artifacts/recommendation_runs/partition-0-cpu-full/`, including model/optimizer checkpoint data and run metadata; it remains a partial update-50 checkpoint, not a completed epoch.
- `upstream-mind/`, `upstream-wenzheng/`, `upstream-gezi/`, `upstream-melt/`, `upstream-msnews-mind/`; `source_lock.json`, `source_manifest.json`, `source_provenance.json`, `dependency_source_lock.json`, and `tokenizer_artifact.json` were left unchanged.

The immutable source-lock JSON files retain historical pointers to some removed status/runtime reports. Those are provenance notes only; no active Python import or pipeline command depends on those report files.

Only source/config/docs and small manifest metadata were reviewed or changed. No data scan/checksum, preprocessing, embedding training, model import/forward, smoke test, benchmark, evaluation, prediction, or recommendation training was performed. No runtime PASS is claimed for this cleanup.

## Follow-up: retain the current Kaggle training handoff

Date: 2026-11-13. The earlier cleanup entry above accurately records that `deployment/kaggle/` was removed on 2026-11-12. The active project later reintroduced the current `deployment/kaggle/` source-only runtime ZIP, its manifest, packager, and one training notebook as a local handoff. This is a later change, not a correction to the earlier history. The ZIP excludes MIND data, SQLite, vocabularies, embeddings, checkpoints, and credentials; the notebook is local source and has not been uploaded or run from this review. No Kaggle resource was accessed or modified.
