"""SQLite-backed stage-01 candidate batches; histories/news are stored once."""
import sqlite3
import struct

import numpy as np

from .config import CONFIG
from mind_config import EXECUTION_CONFIG

PAD_ID = int(CONFIG.pad_id)
UNK_ID = int(CONFIG.unk_id)


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


class Stage01Store:
    def __init__(self, db_path):
        self.db_path = str(db_path)

    def uid_embedding_rows(self):
        """Number of UID rows selected by test_start + min_uid_count.

        This reimplementation's stable mapping reserves PAD=0 and UNK=1, then
        ranks raw UIDs by the source-derived weighted count. Rows passing the
        source threshold must form one contiguous prefix after the two specials.
        """
        conn = sqlite3.connect("file:{}?mode=ro".format(self.db_path), uri=True)
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
        return 2 + len(eligible)

    def vocab_sizes(self):
        conn = sqlite3.connect("file:{}?mode=ro".format(self.db_path), uri=True)
        try:
            sizes = {r[0]: int(r[1]) for r in conn.execute("SELECT field,COUNT(*) FROM vocab GROUP BY field")}
        finally:
            conn.close()
        sizes["uid"] = self.uid_embedding_rows()
        return sizes

    def batch(self, split, limit=EXECUTION_CONFIG.default_candidate_batch_size, offset=0):
        """Return (features, labels, metadata); labels never enter features."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
          SELECT c.id,c.did,c.position,i.id,i.impression_id,i.uid,i.history_ids,
                 n.cat_id,n.sub_cat_id,n.title_ids,n.abstract_ids,n.body_ids,
                 n.title_entity_ids,n.title_entity_types,
                 n.abstract_entity_ids,n.abstract_entity_types,l.click,l.labeled
          FROM candidates c JOIN impressions i ON i.id=c.impression_id
          LEFT JOIN news n ON n.did=c.did JOIN labels l ON l.candidate_id=c.id
          WHERE i.split=? ORDER BY i.line_no,c.position LIMIT ? OFFSET ?
        """, (split, int(limit), int(offset))).fetchall()
        if not rows:
            conn.close()
            return None, None, None
        # Fetch each distinct historical news row once for this batch.
        hist_by_imp = {}
        all_dids = set()
        for row in rows:
            cid, did, pos, imp_id, imp, uid, hist_blob = row[:7]
            hist = _ids(hist_blob, CONFIG.max_history)
            hist_by_imp[imp_id] = hist
            all_dids.add(int(did))
            all_dids.update(int(x) for x in hist if x)
        lookup = {}
        did_list = list(all_dids)
        for start in range(0, len(did_list), 800):
            part = did_list[start:start + 800]
            marks = ",".join("?" for _ in part)
            query = ("SELECT did,cat_id,sub_cat_id,title_ids,abstract_ids,body_ids,"
                     "title_entity_ids,title_entity_types,abstract_entity_ids,abstract_entity_types "
                     "FROM news WHERE did IN (" + marks + ")")
            for rec in conn.execute(query, part):
                lookup[rec[0]] = rec
        conn.close()

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
          "cur_body_tokens": _padding_array((b, CONFIG.max_body_tokens)),
          "his_body_tokens": _padding_array((b, CONFIG.max_history_bodies, CONFIG.max_body_tokens)),
          "body_available": False,
        }
        labels = np.zeros((b,), np.int32)
        labeled = np.zeros((b,), np.bool_)
        uid_embedding_rows = self.uid_embedding_rows()
        metadata = []
        for j, row in enumerate(rows):
            cid,did,pos,imp_id,imp,uid,hist_blob,cat,subcat,title,abstract,body,te,tt,ae,at,click,is_labeled = row
            f["uid"][j] = uid_embedding_id(uid, uid_embedding_rows, CONFIG.unk_id)
            f["did"][j] = did
            f["cat"][j], f["sub_cat"][j] = cat or UNK_ID, subcat or UNK_ID
            f["cur_title_tokens"][j] = _ids(title, CONFIG.max_title_tokens)
            f["cur_abstract_tokens"][j] = _ids(abstract, CONFIG.max_abstract_tokens)
            f["cur_body_tokens"][j] = _ids(body, CONFIG.max_body_tokens)
            f["body_available"] = f["body_available"] or body is not None
            f["title_entities"][j] = _ids(te, CONFIG.max_title_entities)
            f["title_entity_types"][j] = _ids(tt, CONFIG.max_title_entities)
            f["abstract_entities"][j] = _ids(ae, CONFIG.max_abstract_entities)
            f["abstract_entity_types"][j] = _ids(at, CONFIG.max_abstract_entities)
            hist = hist_by_imp[imp_id]
            f["his_id"][j] = hist
            for k, hdid in enumerate(hist):
                if hdid == PAD_ID:
                    continue
                hrow = lookup.get(hdid)
                if hrow is None:
                    continue
                _, hcat, hsubcat, ht, ha, hb, hte, htt, hae, hat = hrow
                f["his_cats"][j,k], f["his_sub_cats"][j,k] = hcat, hsubcat
                if k < CONFIG.max_history_titles:
                    f["his_title_tokens"][j,k] = _ids(ht, CONFIG.max_title_tokens)
                    ent_pos = k * CONFIG.max_history_title_entities_per_news
                    f["his_title_merge_entities"][j,ent_pos:ent_pos+2] = _ids(hte, 2)
                    f["his_title_merge_entity_types"][j,ent_pos:ent_pos+2] = _ids(htt, 2)
                if k < CONFIG.max_history_abstracts:
                    f["his_abstract_tokens"][j,k] = _ids(ha, CONFIG.max_abstract_tokens)
                    ent_pos = k * CONFIG.max_history_abstract_entities_per_news
                    f["his_abstract_merge_entities"][j,ent_pos:ent_pos+2] = _ids(hae, 2)
                    f["his_abstract_merge_entity_types"][j,ent_pos:ent_pos+2] = _ids(hat, 2)
                if k < CONFIG.max_history_bodies:
                    f["his_body_tokens"][j,k] = _ids(hb, CONFIG.max_body_tokens)
                    f["body_available"] = f["body_available"] or hb is not None
            labels[j] = int(click or 0)
            labeled[j] = bool(is_labeled)
            metadata.append({"candidate_id": cid, "impression_id": imp, "position": pos,
                             "uid_vocab_id": uid,
                             "uid_embedding_id": int(f["uid"][j])})
        return f, {"click": labels, "labeled": labeled}, metadata
