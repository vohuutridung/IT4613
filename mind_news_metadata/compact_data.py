"""Memmap-backed batch store for compact MIND prepared bundles."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from mind_data_pipeline.config import CONFIG
from mind_config import EXECUTION_CONFIG, TRAINING_CONFIG
from .batch_data import (
    PAD_ID,
    UNK_ID,
)


class CompactNewsBatchStore:
    def __init__(self, prepared_root, **_):
        self.root = Path(prepared_root).resolve(strict=True)
        self.db_path = str(self.root)
        compact = self.root / "compact"
        self.manifest = json.loads(
            (compact / "compact_manifest.json").read_text(encoding="utf-8"))
        self.layout = json.loads(
            (compact / "layout.json").read_text(encoding="utf-8"))
        self.compact = compact
        self._behaviors = {}
        self._load_news()
        self._load_did_tokens()

    def _load_news(self):
        root = self.compact / "news"
        names = (
            "present", "source_split", "cat", "sub_cat", "title", "abstract",
            "title_entities", "title_entity_types",
            "abstract_entities", "abstract_entity_types")
        self.news = {
            name: np.load(root / (name + ".npy"), mmap_mode="r")
            for name in names
        }

    def _load_did_tokens(self):
        tokens = []
        with (self.root / "vocab/did.tsv").open(encoding="utf-8") as stream:
            for line in stream:
                index, token, *_ = line.rstrip("\n").split("\t")
                if int(index) != len(tokens):
                    raise ValueError("DID vocabulary IDs are not contiguous")
                tokens.append(token)
        self.did_tokens = tokens

    def _split(self, split):
        if split not in self._behaviors:
            meta = self.layout["splits"][split]
            root = self.compact / "behaviors" / split
            histories = int(meta["history_values"])
            candidates = int(meta["candidates"])
            self._behaviors[split] = {
                "meta": meta,
                "uid": np.load(root / "uid.npy", mmap_mode="r"),
                "impression_id": np.load(
                    root / "impression_id.npy", mmap_mode="r"),
                "history_offsets": np.load(
                    root / "history_offsets.npy", mmap_mode="r"),
                "candidate_offsets": np.load(
                    root / "candidate_offsets.npy", mmap_mode="r"),
                "history_did": np.memmap(
                    root / "history_did.i32", mode="r", dtype="<i4",
                    shape=(histories,)),
                "candidate_did": np.memmap(
                    root / "candidate_did.i32", mode="r", dtype="<i4",
                    shape=(candidates,)),
                "click": np.memmap(
                    root / "click.i8", mode="r", dtype="i1",
                    shape=(candidates,)),
            }
        return self._behaviors[split]

    def uid_embedding_rows(self):
        return int(self.layout["vocab"]["uid_embedding_rows"])

    def vocab_sizes(self):
        result = {
            key: int(value)
            for key, value in self.layout["vocab"]["sizes"].items()
        }
        result["uid"] = self.uid_embedding_rows()
        return result

    def impression_ids(self, split):
        data = self._split(split)
        base = int(data["meta"]["impression_base"])
        return np.arange(
            base, base + int(data["meta"]["impressions"]), dtype=np.int64)

    def count_candidates(self, split):
        return int(self._split(split)["meta"]["candidates"])

    def candidate_groups(self, split, impression_ids):
        data = self._split(split)
        base = int(data["meta"]["impression_base"])
        result = {}
        for global_id in map(int, impression_ids):
            local = global_id - base
            if not 0 <= local < int(data["meta"]["impressions"]):
                raise ValueError("impression ID is outside {} compact split".format(split))
            start = int(data["candidate_offsets"][local])
            stop = int(data["candidate_offsets"][local + 1])
            result[global_id] = {
                "impression_id": self._decode_impression(data, local),
                "positions": np.arange(stop - start, dtype=np.int64),
                "click": np.asarray(data["click"][start:stop], dtype=np.int32),
                "labeled": np.asarray(data["click"][start:stop] >= 0, dtype=np.bool_),
            }
        return result

    def count_selected_partition(self, impression_ids, partition, seed, split="train"):
        """Exact selected-candidate totals for one train negative partition."""
        if split != "train":
            raise ValueError("negative partition sampling is train-only")
        from .recommendation_protocol import count_partition_from_offsets

        data = self._split(split)
        base = int(data["meta"]["impression_base"])
        n_impressions = int(data["meta"]["impressions"])
        selected = np.asarray(impression_ids, dtype=np.int64) - base
        if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= n_impressions):
            raise ValueError("impression ID is outside {} compact split".format(split))
        offsets = np.array(data["candidate_offsets"], dtype=np.int64, copy=True)
        clicks = np.array(data["click"], dtype=np.int8, copy=True)
        return count_partition_from_offsets(
            clicks, offsets, data["impression_id"], selected, partition, seed, split=split)

    def iter_submission_keys(self, split="test"):
        data = self._split(split)
        for local in range(int(data["meta"]["impressions"])):
            impression = self._decode_impression(data, local)
            start = int(data["candidate_offsets"][local])
            stop = int(data["candidate_offsets"][local + 1])
            for position, index in enumerate(range(start, stop)):
                did = int(data["candidate_did"][index])
                news_id = (self.did_tokens[did]
                           if 0 <= did < len(self.did_tokens)
                           and bool(self.news["present"][did]) else None)
                yield impression, position, news_id

    @staticmethod
    def _decode_impression(data, local):
        return bytes(data["impression_id"][local]).rstrip(b"\0").decode("utf-8")

    def _candidate_selection(self, split, limit, offset, after_key, impression_ids):
        data = self._split(split)
        offsets = data["candidate_offsets"]
        if impression_ids is not None:
            base = int(data["meta"]["impression_base"])
            candidate_parts, impression_parts = [], []
            for global_id in map(int, impression_ids):
                local = global_id - base
                if not 0 <= local < int(data["meta"]["impressions"]):
                    raise ValueError("impression ID outside compact split")
                start, stop = int(offsets[local]), int(offsets[local + 1])
                candidate_parts.append(np.arange(start, stop, dtype=np.int64))
                impression_parts.append(
                    np.full(stop - start, local, dtype=np.int64))
            if not candidate_parts:
                return np.empty(0, np.int64), np.empty(0, np.int64)
            return np.concatenate(candidate_parts), np.concatenate(impression_parts)
        if after_key is not None:
            line_no, position = map(int, after_key)
            if line_no <= 0:
                start = 0
            else:
                local = line_no - 1
                if local >= int(data["meta"]["impressions"]):
                    return np.empty(0, np.int64), np.empty(0, np.int64)
                start = int(offsets[local]) + position + 1
        else:
            start = int(offset)
        stop = min(start + int(limit), int(data["meta"]["candidates"]))
        candidate_indices = np.arange(start, stop, dtype=np.int64)
        impression_indices = np.searchsorted(
            offsets, candidate_indices, side="right") - 1
        return candidate_indices, impression_indices.astype(np.int64)

    def batch(self, split, limit=EXECUTION_CONFIG.default_candidate_batch_size,
              offset=0, after_key=None, impression_ids=None, partition=None,
              partition_seed=TRAINING_CONFIG["partition_base_seed"],
              profile_timings=None, include_metadata=True):
        started = time.perf_counter()
        data = self._split(split)
        candidate_indices, impression_indices = self._candidate_selection(
            split, limit, offset, after_key, impression_ids)
        if partition is not None and len(candidate_indices):
            from .recommendation_protocol import partition_for_impression
            keep_all = np.zeros(len(candidate_indices), dtype=np.bool_)
            cursor = 0
            while cursor < len(candidate_indices):
                local = int(impression_indices[cursor])
                end = cursor + 1
                while (end < len(candidate_indices)
                       and int(impression_indices[end]) == local):
                    end += 1
                clicks = np.asarray(
                    data["click"][candidate_indices[cursor:end]], dtype=np.int32)
                if np.any(clicks < 0):
                    raise ValueError("training partition encountered an unlabeled candidate")
                positions = candidate_indices[cursor:end] - int(
                    data["candidate_offsets"][local])
                mask = partition_for_impression(
                    clicks, positions, self._decode_impression(data, local),
                    partition, base_seed=partition_seed, split=split)
                keep_all[cursor:end] = mask
                cursor = end
            candidate_indices = candidate_indices[keep_all]
            impression_indices = impression_indices[keep_all]
        if not len(candidate_indices):
            return None, None, None

        count = len(candidate_indices)
        dids = np.asarray(data["candidate_did"][candidate_indices], dtype=np.int32)
        raw_uids = np.asarray(data["uid"][impression_indices], dtype=np.int32)
        uid_rows = self.uid_embedding_rows()
        f = {"uid": np.where((raw_uids >= 0) & (raw_uids < uid_rows), raw_uids, UNK_ID),
             "did": dids}
        n_news = len(self.news["present"])
        safe = np.clip(dids, 0, n_news - 1)
        valid = (dids >= 0) & (dids < n_news) & self.news["present"][safe]
        for target, source in (("cat", "cat"), ("sub_cat", "sub_cat"),
                               ("cur_title_tokens", "title"), ("cur_abstract_tokens", "abstract"),
                               ("title_entities", "title_entities"),
                               ("title_entity_types", "title_entity_types"),
                               ("abstract_entities", "abstract_entities"),
                               ("abstract_entity_types", "abstract_entity_types")):
            values = np.array(self.news[source][safe], dtype=np.int32, copy=True)
            if target in ("cat", "sub_cat"):
                values[values == 0] = UNK_ID
            values[~valid] = PAD_ID
            f[target] = values

        # Gather each impression's history once, then expand to its candidates.
        # No Python loop over candidates, history items or feature slots.
        unique, inverse = np.unique(impression_indices, return_inverse=True)
        starts = np.asarray(data["history_offsets"][unique])
        lengths = np.minimum(np.asarray(data["history_offsets"][unique + 1]) - starts,
                             int(CONFIG.max_history))
        positions = np.arange(int(CONFIG.max_history))[None, :]
        history = np.zeros((len(unique), int(CONFIG.max_history)), dtype=np.int32)
        history_valid = positions < lengths[:, None]
        indices = starts[:, None] + positions
        history[history_valid] = data["history_did"][indices[history_valid]]
        f["his_id"] = history[inverse]
        safe_history = np.clip(history, 0, n_news - 1)
        present = ((history != PAD_ID) & (history >= 0) & (history < n_news)
                   & self.news["present"][safe_history])
        for target, source in (("his_cats", "cat"), ("his_sub_cats", "sub_cat")):
            values = np.array(self.news[source][safe_history], dtype=np.int32, copy=True)
            values[~present] = PAD_ID
            f[target] = values[inverse]
        for branch, width in (("title", int(CONFIG.max_history_titles)),
                              ("abstract", int(CONFIG.max_history_abstracts))):
            history_ids = safe_history[:, :width]
            mask = present[:, :width]
            values = np.array(self.news[branch][history_ids], dtype=np.int32, copy=True)
            values[~mask] = PAD_ID
            f["his_" + branch + "_tokens"] = values[inverse]
            for suffix in ("entities", "entity_types"):
                values = np.array(self.news[branch + "_" + suffix][history_ids, :2],
                                  dtype=np.int32, copy=True)
                values[~mask] = PAD_ID
                target = "his_" + branch + "_merge_" + suffix
                f[target] = values.reshape(len(unique), width * 2)[inverse]

        raw_click = np.asarray(data["click"][candidate_indices], dtype=np.int32)
        labels = np.maximum(raw_click, 0)
        labeled = raw_click >= 0
        metadata = None
        if include_metadata:
            impression_base = int(data["meta"]["impression_base"])
            candidate_base = int(data["meta"]["candidate_base"])
            tokens = {int(local): self._decode_impression(data, int(local)) for local in unique}
            positions = candidate_indices - data["candidate_offsets"][impression_indices]
            metadata = [{
                "candidate_id": candidate_base + int(candidate_indices[i]),
                "impression_id": tokens[int(impression_indices[i])],
                "position": int(positions[i]), "line_no": int(impression_indices[i]) + 1,
                "split": split, "news_id": self.did_tokens[int(dids[i])] if valid[i] else None,
                "uid_vocab_id": int(raw_uids[i]), "uid_embedding_id": int(f["uid"][i]),
            } for i in range(count)]
        if profile_timings is not None:
            profile_timings.update({
                "storage_format": "compact_memmap",
                "candidate_rows_selected": count,
                "impressions_selected": len(unique),
                "total_seconds": time.perf_counter() - started,
            })
        return f, {"click": labels, "labeled": labeled}, metadata
