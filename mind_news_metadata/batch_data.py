"""No-body 15-field batch view over the prepared MIND-large SQLite artifact."""
import sqlite3
import struct
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from mind_data_pipeline.config import CONFIG
from mind_config import EXECUTION_CONFIG, TRAINING_CONFIG

PAD_ID = int(CONFIG.pad_id)
UNK_ID = int(CONFIG.unk_id)


def connect_readonly(db_path):
    """Open an immutable input DB without needing WAL/SHM writes beside it."""
    path = Path(db_path).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError("SQLite input is not a regular file: {}".format(path))
    # as_uri percent-encodes spaces, '#', '%', and other URI-sensitive chars.
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _padding_array(shape, dtype=np.int32):
    return np.full(shape, PAD_ID, dtype=dtype)


def _ids(blob, n):
    if blob is None or len(blob) == 0:
        return [PAD_ID] * n
    vals = list(struct.unpack("<{}I".format(len(blob) // 4), blob))
    return vals[:n] + [PAD_ID] * max(0, n - len(vals))


def uid_embedding_id(vocab_id, embedding_rows, unk_id=None):
    """Map a full-vocabulary UID ID to the source-capped table without off-by-one."""
    value, rows = int(vocab_id), int(embedding_rows)
    if unk_id is None:
        unk_id = UNK_ID
    return value if 0 <= value < rows else int(unk_id)


def _decoded_news(row):
    """Decode one DB row once into immutable token/entity ID arrays."""
    did, news_id, cat, subcat, title, abstract, title_entities, title_types, abstract_entities, abstract_types = row
    return {
        "news_id": news_id,
        "cat": cat,
        "sub_cat": subcat,
        "title": np.asarray(_ids(title, CONFIG.max_title_tokens), dtype=np.int32),
        "abstract": np.asarray(_ids(abstract, CONFIG.max_abstract_tokens), dtype=np.int32),
        "title_entities": np.asarray(_ids(title_entities, CONFIG.max_title_entities), dtype=np.int32),
        "title_entity_types": np.asarray(_ids(title_types, CONFIG.max_title_entities), dtype=np.int32),
        "abstract_entities": np.asarray(_ids(abstract_entities, CONFIG.max_abstract_entities), dtype=np.int32),
        "abstract_entity_types": np.asarray(_ids(abstract_types, CONFIG.max_abstract_entities), dtype=np.int32),
    }


def _news_cache_size(value):
    if value is None:
        return 256
    # Reserve 2 KiB per row for Python dict keys/values, strings, and ndarray
    # headers in addition to the exact NumPy buffer sizes. This intentionally
    # overestimates small-object overhead so the 64 MiB cache stays bounded.
    return 2048 + sum(int(array.nbytes) for array in value.values() if isinstance(array, np.ndarray))


def _copy_first_two(target, source):
    target[...] = PAD_ID
    count = min(2, len(source))
    if count:
        target[:count] = source[:count]


class NewsBatchStore:
    def __init__(self, db_path,
                 news_cache_max_bytes=EXECUTION_CONFIG.news_cache_max_bytes,
                 news_cache_max_items=EXECUTION_CONFIG.news_cache_max_items):
        self.db_path = str(db_path)
        self._cached_uid_embedding_rows = None
        self._news_cache = OrderedDict()
        self._news_cache_sizes = {}
        self._news_cache_bytes = 0
        self._news_cache_max_bytes = int(news_cache_max_bytes)
        self._news_cache_max_items = int(news_cache_max_items)
        self._news_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _news_get(self, did):
        did = int(did)
        if did not in self._news_cache:
            self._news_cache_stats["misses"] += 1
            return False, None
        self._news_cache_stats["hits"] += 1
        self._news_cache.move_to_end(did)
        return True, self._news_cache[did]

    def _news_put(self, did, value):
        did = int(did)
        size = _news_cache_size(value)
        if size > self._news_cache_max_bytes or self._news_cache_max_items <= 0:
            return
        if did in self._news_cache:
            self._news_cache_bytes -= self._news_cache_sizes.pop(did)
            del self._news_cache[did]
        self._news_cache[did] = value
        self._news_cache_sizes[did] = size
        self._news_cache_bytes += size
        while (self._news_cache_bytes > self._news_cache_max_bytes
               or len(self._news_cache) > self._news_cache_max_items):
            old_did, _ = self._news_cache.popitem(last=False)
            self._news_cache_bytes -= self._news_cache_sizes.pop(old_did)
            self._news_cache_stats["evictions"] += 1

    def cache_info(self):
        return {"items": len(self._news_cache), "estimated_bytes": self._news_cache_bytes,
                "max_items": self._news_cache_max_items, "max_bytes": self._news_cache_max_bytes,
                **self._news_cache_stats}

    def uid_embedding_rows(self):
        """Number of UID rows selected by test_start + min_uid_count.

        This reimplementation's stable mapping reserves PAD=0 and UNK=1, then
        ranks raw UIDs by the source-derived weighted count. Rows passing the
        source threshold must form one contiguous prefix after the two specials.
        """
        if self._cached_uid_embedding_rows is not None:
            return self._cached_uid_embedding_rows
        conn = connect_readonly(self.db_path)
        try:
            threshold = int(CONFIG.test_start + CONFIG.min_uid_count)
            rows = conn.execute("SELECT id,source_count FROM vocab WHERE field='uid' AND id>=2 ORDER BY id").fetchall()
        finally:
            conn.close()
        eligible = [row_id for row_id, count in rows if count is not None and int(count) >= threshold]
        if eligible and (eligible[0] != 2 or eligible[-1] - eligible[0] + 1 != len(eligible)):
            raise ValueError("UIDs meeting min_uid_count do not form the expected ranked prefix")
        if any(count is not None and int(count) >= threshold for _, count in rows[len(eligible):]):
            raise ValueError("UID vocabulary ordering is inconsistent with the count threshold")
        self._cached_uid_embedding_rows = 2 + len(eligible)
        return self._cached_uid_embedding_rows

    def vocab_sizes(self):
        conn = connect_readonly(self.db_path)
        try:
            sizes = {r[0]: int(r[1]) for r in conn.execute("SELECT field,COUNT(*) FROM vocab GROUP BY field")}
        finally:
            conn.close()
        sizes["uid"] = self.uid_embedding_rows()
        return sizes

    def batch(self, split, limit=EXECUTION_CONFIG.default_candidate_batch_size, offset=0, after_key=None,
              impression_ids=None, partition=None,
              partition_seed=TRAINING_CONFIG["partition_base_seed"],
              profile_timings=None, include_metadata=True):
        """Return (features, labels, metadata); callers may omit unused provenance metadata."""
        batch_started = time.perf_counter()
        cache_before = dict(self._news_cache_stats)
        conn = connect_readonly(self.db_path)
        connected = time.perf_counter()
        query = """
          SELECT c.id,c.did,c.position,i.id,i.impression_id,i.uid,i.history_ids,
                 l.click,l.labeled,i.line_no
          FROM candidates c JOIN impressions i ON i.id=c.impression_id
          JOIN labels l ON l.candidate_id=c.id
          WHERE i.split=?
        """
        params = [split]
        if impression_ids is not None:
            ids = [int(x) for x in impression_ids]
            if not ids:
                conn.close()
                return None, None, None
            query += " AND i.id IN ({})".format(",".join("?" for _ in ids))
            params.extend(ids)
            query += " ORDER BY i.id,c.position"
            rows = conn.execute(query, params).fetchall()
            order_rank = {impression_id: index for index, impression_id in enumerate(ids)}
            rows.sort(key=lambda row: (order_rank[row[3]], row[2]))
        elif after_key is not None:
            line_no, position = map(int, after_key)
            query += " AND (i.line_no>? OR (i.line_no=? AND c.position>?))"
            params.extend([line_no, line_no, position])
            query += " ORDER BY i.line_no,c.position LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(query, params).fetchall()
        else:
            query += " ORDER BY i.line_no,c.position LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])
            rows = conn.execute(query, params).fetchall()
        candidate_query_done = time.perf_counter()
        candidate_rows_fetched = len(rows)
        if partition is not None and rows:
            from .recommendation_protocol import partition_for_impression
            grouped = {}
            for row in rows:
                grouped.setdefault(row[3], []).append(row)
            selected_ids = set()
            for impression_key, group in grouped.items():
                group.sort(key=lambda x: x[2])
                if not all(bool(row[8]) for row in group):
                    conn.close()
                    raise ValueError("training partition encountered an unlabeled candidate")
                keep = partition_for_impression(
                    [int(row[7]) for row in group], [int(row[2]) for row in group],
                    group[0][4], partition, base_seed=partition_seed, split=split)
                selected_ids.update(row[0] for row, use in zip(group, keep) if use)
            rows = [row for row in rows if row[0] in selected_ids]
        partition_done = time.perf_counter()
        if not rows:
            conn.close()
            if profile_timings is not None:
                profile_timings.update({
                    "connect_seconds": connected - batch_started,
                    "candidate_query_fetch_seconds": candidate_query_done - connected,
                    "partition_seconds": partition_done - candidate_query_done,
                    "news_lookup_seconds": 0.0,
                    "news_cache_lookup_seconds": 0.0,
                    "news_query_fetch_seconds": 0.0,
                    "news_decode_seconds": 0.0,
                    "history_decode_seconds": 0.0,
                    "history_materialization_seconds": 0.0,
                    "array_allocation_seconds": 0.0,
                    "candidate_feature_materialization_seconds": 0.0,
                    "feature_materialization_seconds": 0.0,
                    "total_seconds": time.perf_counter() - batch_started,
                    "candidate_rows_fetched": candidate_rows_fetched,
                    "candidate_rows_selected": 0,
                    "impressions_selected": 0,
                    "history_slots_decoded": 0,
                    "history_nonpadding_ids_decoded": 0,
                    "unique_news_ids_requested": 0,
                    "news_rows_decoded": 0,
                    "news_query_count": 0,
                    "news_cache_hits": self._news_cache_stats["hits"] - cache_before["hits"],
                    "news_cache_misses": self._news_cache_stats["misses"] - cache_before["misses"],
                    "news_cache_evictions": self._news_cache_stats["evictions"] - cache_before["evictions"],
                })
            return None, None, None

        # Decode each impression history once. Candidate rows in the same
        # impression share the ordered history list.
        history_decode_started = time.perf_counter()
        hist_by_imp = {}
        all_dids = {}
        for row in rows:
            did, imp_id, hist_blob = int(row[1]), row[3], row[6]
            all_dids.setdefault(did, None)
            if imp_id not in hist_by_imp:
                hist = tuple(_ids(hist_blob, CONFIG.max_history))
                hist_by_imp[imp_id] = hist
                for hdid in hist:
                    if hdid:
                        all_dids.setdefault(int(hdid), None)
        history_decode_seconds = time.perf_counter() - history_decode_started
        history_slots_decoded = sum(len(history) for history in hist_by_imp.values())
        history_nonpadding_ids_decoded = sum(
            sum(1 for did in history if did) for history in hist_by_imp.values())

        lookup = {}
        news_cache_lookup_started = time.perf_counter()
        missing_dids = []
        for did in all_dids:
            found, value = self._news_get(did)
            if found:
                lookup[did] = value
            else:
                missing_dids.append(did)
        cache_lookup_done = time.perf_counter()
        news_query_fetch_seconds = news_decode_seconds = 0.0
        news_query_count = news_rows_decoded = 0
        for start in range(0, len(missing_dids), 800):
            part = missing_dids[start:start + 800]
            marks = ",".join("?" for _ in part)
            query = ("SELECT did,news_id,cat_id,sub_cat_id,title_ids,abstract_ids,"
                     "title_entity_ids,title_entity_types,abstract_entity_ids,abstract_entity_types "
                     "FROM news WHERE did IN (" + marks + ")")
            fetch_started = time.perf_counter()
            news_rows = conn.execute(query, part).fetchall()
            news_query_count += 1
            news_rows_decoded += len(news_rows)
            news_query_fetch_seconds += time.perf_counter() - fetch_started
            decode_started = time.perf_counter()
            for rec in news_rows:
                decoded = _decoded_news(rec)
                lookup[int(rec[0])] = decoded
                self._news_put(rec[0], decoded)
            news_decode_seconds += time.perf_counter() - decode_started
        for did in missing_dids:
            if did not in lookup:
                lookup[did] = None
                self._news_put(did, None)
        conn.close()
        news_lookup_done = time.perf_counter()

        b = len(rows)
        f = {
          "uid": _padding_array((b,)), "did": _padding_array((b,)),
          "his_id": _padding_array((b, CONFIG.max_history)),
          "cat": _padding_array((b,)), "sub_cat": _padding_array((b,)),
          "title_entities": _padding_array((b, CONFIG.max_title_entities)),
          "title_entity_types": _padding_array((b, CONFIG.max_title_entities)),
          "abstract_entities": _padding_array((b, CONFIG.max_abstract_entities)),
          "abstract_entity_types": _padding_array((b, CONFIG.max_abstract_entities)),
          "his_title_merge_entities": _padding_array((b, CONFIG.max_history_titles * CONFIG.max_history_title_entities_per_news)),
          "his_title_merge_entity_types": _padding_array((b, CONFIG.max_history_titles * CONFIG.max_history_title_entities_per_news)),
          "his_abstract_merge_entities": _padding_array((b, CONFIG.max_history_abstracts * CONFIG.max_history_abstract_entities_per_news)),
          "his_abstract_merge_entity_types": _padding_array((b, CONFIG.max_history_abstracts * CONFIG.max_history_abstract_entities_per_news)),
          "his_cats": _padding_array((b, CONFIG.max_history)),
          "his_sub_cats": _padding_array((b, CONFIG.max_history)),
          "cur_title_tokens": _padding_array((b, CONFIG.max_title_tokens)),
          "his_title_tokens": _padding_array((b, CONFIG.max_history_titles, CONFIG.max_title_tokens)),
          "cur_abstract_tokens": _padding_array((b, CONFIG.max_abstract_tokens)),
          "his_abstract_tokens": _padding_array((b, CONFIG.max_history_abstracts, CONFIG.max_abstract_tokens)),
        }
        labels = np.zeros((b,), np.int32)
        labeled = np.zeros((b,), np.bool_)
        uid_embedding_rows = self.uid_embedding_rows()
        metadata = [] if include_metadata else None
        arrays_allocated = time.perf_counter()

        # Materialize the raw history-side ID tensors once per impression.
        # These cached arrays do not depend on candidate query or model weights.
        history_materialization_started = time.perf_counter()
        history_features = {}
        for imp_id, hist in hist_by_imp.items():
            h = {
                "his_id": np.asarray(hist, dtype=np.int32),
                "his_cats": _padding_array((CONFIG.max_history,)),
                "his_sub_cats": _padding_array((CONFIG.max_history,)),
                "his_title_tokens": _padding_array((CONFIG.max_history_titles, CONFIG.max_title_tokens)),
                "his_title_merge_entities": _padding_array((CONFIG.max_history_titles * CONFIG.max_history_title_entities_per_news,)),
                "his_title_merge_entity_types": _padding_array((CONFIG.max_history_titles * CONFIG.max_history_title_entities_per_news,)),
                "his_abstract_tokens": _padding_array((CONFIG.max_history_abstracts, CONFIG.max_abstract_tokens)),
                "his_abstract_merge_entities": _padding_array((CONFIG.max_history_abstracts * CONFIG.max_history_abstract_entities_per_news,)),
                "his_abstract_merge_entity_types": _padding_array((CONFIG.max_history_abstracts * CONFIG.max_history_abstract_entities_per_news,)),
            }
            for k, hdid in enumerate(hist):
                if hdid == PAD_ID:
                    continue
                hrow = lookup.get(int(hdid))
                if hrow is None:
                    continue
                h["his_cats"][k], h["his_sub_cats"][k] = hrow["cat"], hrow["sub_cat"]
                if k < CONFIG.max_history_titles:
                    h["his_title_tokens"][k] = hrow["title"]
                    ent_pos = k * CONFIG.max_history_title_entities_per_news
                    _copy_first_two(h["his_title_merge_entities"][ent_pos:ent_pos + 2], hrow["title_entities"])
                    _copy_first_two(h["his_title_merge_entity_types"][ent_pos:ent_pos + 2], hrow["title_entity_types"])
                if k < CONFIG.max_history_abstracts:
                    h["his_abstract_tokens"][k] = hrow["abstract"]
                    ent_pos = k * CONFIG.max_history_abstract_entities_per_news
                    _copy_first_two(h["his_abstract_merge_entities"][ent_pos:ent_pos + 2], hrow["abstract_entities"])
                    _copy_first_two(h["his_abstract_merge_entity_types"][ent_pos:ent_pos + 2], hrow["abstract_entity_types"])
            history_features[imp_id] = h
        history_materialization_seconds = time.perf_counter() - history_materialization_started

        candidate_materialization_started = time.perf_counter()
        # Candidate rows are ordered contiguously by impression. Broadcast
        # each already-materialized history block once per impression instead
        # of copying every history field in Python for every candidate.
        history_ranges = []
        closed_impressions = set()
        active_impression = None
        active_start = 0
        for j, row in enumerate(rows):
            cid,did,pos,imp_id,imp,uid,hist_blob,click,is_labeled,line_no = row
            if imp_id != active_impression:
                if active_impression is not None:
                    history_ranges.append((active_impression, active_start, j))
                    closed_impressions.add(active_impression)
                if imp_id in closed_impressions:
                    raise ValueError("candidate rows are not contiguous by impression")
                active_impression, active_start = imp_id, j
            f["uid"][j] = uid_embedding_id(uid, uid_embedding_rows, CONFIG.unk_id)
            f["did"][j] = did
            f["cat"][j], f["sub_cat"][j] = UNK_ID, UNK_ID
            current = lookup.get(int(did))
            if current is not None:
                f["cat"][j], f["sub_cat"][j] = current["cat"] or UNK_ID, current["sub_cat"] or UNK_ID
                f["cur_title_tokens"][j] = current["title"]
                f["cur_abstract_tokens"][j] = current["abstract"]
                f["title_entities"][j] = current["title_entities"]
                f["title_entity_types"][j] = current["title_entity_types"]
                f["abstract_entities"][j] = current["abstract_entities"]
                f["abstract_entity_types"][j] = current["abstract_entity_types"]
            labels[j] = int(click or 0)
            labeled[j] = bool(is_labeled)
            if include_metadata:
                metadata.append({"candidate_id": cid, "impression_id": imp, "position": pos,
                                 "line_no": line_no, "split": split,
                                 "news_id": current["news_id"] if current is not None else None,
                                 "uid_vocab_id": uid,
                                 "uid_embedding_id": int(f["uid"][j])})
        if active_impression is not None:
            history_ranges.append((active_impression, active_start, len(rows)))
        for imp_id, start, stop in history_ranges:
            for key, value in history_features[imp_id].items():
                f[key][start:stop] = value
        candidate_materialization_seconds = time.perf_counter() - candidate_materialization_started
        if profile_timings is not None:
            profile_timings.update({
                "connect_seconds": connected - batch_started,
                "candidate_query_fetch_seconds": candidate_query_done - connected,
                "partition_seconds": partition_done - candidate_query_done,
                "news_lookup_seconds": news_lookup_done - partition_done,
                "news_cache_lookup_seconds": cache_lookup_done - news_cache_lookup_started,
                "news_query_fetch_seconds": news_query_fetch_seconds,
                "news_decode_seconds": news_decode_seconds,
                "history_decode_seconds": history_decode_seconds,
                "history_materialization_seconds": history_materialization_seconds,
                "array_allocation_seconds": arrays_allocated - news_lookup_done,
                "candidate_feature_materialization_seconds": candidate_materialization_seconds,
                "feature_materialization_seconds": history_materialization_seconds + candidate_materialization_seconds,
                "total_seconds": time.perf_counter() - batch_started,
                "candidate_rows_fetched": candidate_rows_fetched,
                "candidate_rows_selected": len(rows),
                "impressions_selected": len(hist_by_imp),
                "history_slots_decoded": history_slots_decoded,
                "history_nonpadding_ids_decoded": history_nonpadding_ids_decoded,
                "unique_news_ids_requested": len(all_dids),
                "news_rows_decoded": news_rows_decoded,
                "news_query_count": news_query_count,
                "news_cache_hits": self._news_cache_stats["hits"] - cache_before["hits"],
                "news_cache_misses": self._news_cache_stats["misses"] - cache_before["misses"],
                "news_cache_evictions": self._news_cache_stats["evictions"] - cache_before["evictions"],
                "news_cache_items": len(self._news_cache),
                "news_cache_bytes_estimated": self._news_cache_bytes,
            })
        return f, {"click": labels, "labeled": labeled}, metadata
