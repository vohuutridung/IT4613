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
    _copy_first_two,
    _padding_array,
    uid_embedding_id,
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
        f = {
            "uid": _padding_array((count,)), "did": _padding_array((count,)),
            "his_id": _padding_array((count, int(CONFIG.max_history))),
            "cat": _padding_array((count,)), "sub_cat": _padding_array((count,)),
            "title_entities": _padding_array((count, int(CONFIG.max_title_entities))),
            "title_entity_types": _padding_array((count, int(CONFIG.max_title_entities))),
            "abstract_entities": _padding_array((count, int(CONFIG.max_abstract_entities))),
            "abstract_entity_types": _padding_array((count, int(CONFIG.max_abstract_entities))),
            "his_title_merge_entities": _padding_array(
                (count, int(CONFIG.max_history_titles * CONFIG.max_history_title_entities_per_news))),
            "his_title_merge_entity_types": _padding_array(
                (count, int(CONFIG.max_history_titles * CONFIG.max_history_title_entities_per_news))),
            "his_abstract_merge_entities": _padding_array(
                (count, int(CONFIG.max_history_abstracts * CONFIG.max_history_abstract_entities_per_news))),
            "his_abstract_merge_entity_types": _padding_array(
                (count, int(CONFIG.max_history_abstracts * CONFIG.max_history_abstract_entities_per_news))),
            "his_cats": _padding_array((count, int(CONFIG.max_history))),
            "his_sub_cats": _padding_array((count, int(CONFIG.max_history))),
            "cur_title_tokens": _padding_array((count, int(CONFIG.max_title_tokens))),
            "his_title_tokens": _padding_array(
                (count, int(CONFIG.max_history_titles), int(CONFIG.max_title_tokens))),
            "cur_abstract_tokens": _padding_array((count, int(CONFIG.max_abstract_tokens))),
            "his_abstract_tokens": _padding_array(
                (count, int(CONFIG.max_history_abstracts), int(CONFIG.max_abstract_tokens))),
        }
        labels = np.zeros(count, np.int32)
        labeled = np.zeros(count, np.bool_)
        metadata = [] if include_metadata else None
        history_features = {}
        for local in dict.fromkeys(map(int, impression_indices)):
            h_start = int(data["history_offsets"][local])
            h_stop = int(data["history_offsets"][local + 1])
            history = np.asarray(
                data["history_did"][h_start:h_stop], dtype=np.int32)
            padded = _padding_array((int(CONFIG.max_history),))
            padded[:len(history)] = history
            values = {
                "his_id": padded,
                "his_cats": _padding_array((int(CONFIG.max_history),)),
                "his_sub_cats": _padding_array((int(CONFIG.max_history),)),
                "his_title_tokens": _padding_array(
                    (int(CONFIG.max_history_titles), int(CONFIG.max_title_tokens))),
                "his_title_merge_entities": _padding_array(
                    (int(CONFIG.max_history_titles * CONFIG.max_history_title_entities_per_news),)),
                "his_title_merge_entity_types": _padding_array(
                    (int(CONFIG.max_history_titles * CONFIG.max_history_title_entities_per_news),)),
                "his_abstract_tokens": _padding_array(
                    (int(CONFIG.max_history_abstracts), int(CONFIG.max_abstract_tokens))),
                "his_abstract_merge_entities": _padding_array(
                    (int(CONFIG.max_history_abstracts * CONFIG.max_history_abstract_entities_per_news),)),
                "his_abstract_merge_entity_types": _padding_array(
                    (int(CONFIG.max_history_abstracts * CONFIG.max_history_abstract_entities_per_news),)),
            }
            for index, did in enumerate(history):
                did = int(did)
                if did == PAD_ID or did >= len(self.news["present"]) or not self.news["present"][did]:
                    continue
                values["his_cats"][index] = self.news["cat"][did]
                values["his_sub_cats"][index] = self.news["sub_cat"][did]
                if index < int(CONFIG.max_history_titles):
                    values["his_title_tokens"][index] = self.news["title"][did]
                    pos = index * int(CONFIG.max_history_title_entities_per_news)
                    _copy_first_two(
                        values["his_title_merge_entities"][pos:pos + 2],
                        self.news["title_entities"][did])
                    _copy_first_two(
                        values["his_title_merge_entity_types"][pos:pos + 2],
                        self.news["title_entity_types"][did])
                if index < int(CONFIG.max_history_abstracts):
                    values["his_abstract_tokens"][index] = self.news["abstract"][did]
                    pos = index * int(CONFIG.max_history_abstract_entities_per_news)
                    _copy_first_two(
                        values["his_abstract_merge_entities"][pos:pos + 2],
                        self.news["abstract_entities"][did])
                    _copy_first_two(
                        values["his_abstract_merge_entity_types"][pos:pos + 2],
                        self.news["abstract_entity_types"][did])
            history_features[local] = values

        impression_base = int(data["meta"]["impression_base"])
        candidate_base = int(data["meta"]["candidate_base"])
        uid_rows = self.uid_embedding_rows()
        for row_index, (candidate_index, local) in enumerate(
                zip(candidate_indices, impression_indices)):
            candidate_index, local = int(candidate_index), int(local)
            did = int(data["candidate_did"][candidate_index])
            raw_uid = int(data["uid"][local])
            f["uid"][row_index] = uid_embedding_id(raw_uid, uid_rows, UNK_ID)
            f["did"][row_index] = did
            if 0 <= did < len(self.news["present"]) and self.news["present"][did]:
                f["cat"][row_index] = int(self.news["cat"][did]) or UNK_ID
                f["sub_cat"][row_index] = int(self.news["sub_cat"][did]) or UNK_ID
                f["cur_title_tokens"][row_index] = self.news["title"][did]
                f["cur_abstract_tokens"][row_index] = self.news["abstract"][did]
                for key in (
                        "title_entities", "title_entity_types",
                        "abstract_entities", "abstract_entity_types"):
                    f[key][row_index] = self.news[key][did]
                news_id = self.did_tokens[did]
            else:
                news_id = None
            for key, value in history_features[local].items():
                f[key][row_index] = value
            raw_click = int(data["click"][candidate_index])
            labels[row_index] = max(0, raw_click)
            labeled[row_index] = raw_click >= 0
            if include_metadata:
                position = candidate_index - int(data["candidate_offsets"][local])
                metadata.append({
                    "candidate_id": candidate_base + candidate_index,
                    "impression_id": self._decode_impression(data, local),
                    "position": int(position),
                    "line_no": local + 1,
                    "split": split,
                    "news_id": news_id,
                    "uid_vocab_id": raw_uid,
                    "uid_embedding_id": int(f["uid"][row_index]),
                })
        if profile_timings is not None:
            profile_timings.update({
                "storage_format": "compact_memmap",
                "candidate_rows_selected": count,
                "impressions_selected": len(history_features),
                "total_seconds": time.perf_counter() - started,
            })
        return f, {"click": labels, "labeled": labeled}, metadata
