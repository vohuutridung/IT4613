"""PyTorch field representation encoder for the 15-field variant."""
from pathlib import Path

import torch
from torch import nn

from .encoder_config import ENCODER_CONFIG
from mind_config import TRAINING_CONFIG


def make_linear(in_features, out_features, *, bias=True, trainable=True):
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer.requires_grad_(trainable)


def masked_weights(scores, mask):
    mask = mask.bool()
    weights = scores.float().masked_fill(~mask, ENCODER_CONFIG.masked_score_floor)
    weights = weights.softmax(dim=-1) * mask
    total = weights.sum(dim=-1, keepdim=True)
    return weights / total.masked_fill(total == 0, 1)


class MaskedAttention(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
        width = ENCODER_CONFIG.attention_hidden
        self.proj = make_linear(ENCODER_CONFIG.embedding_dim, width, trainable=trainable)
        self.activation = getattr(torch, ENCODER_CONFIG.attention_activation)
        self.score = make_linear(width, 1, bias=False, trainable=trainable)

    def forward(self, values, mask):
        scores = self.score(self.activation(self.proj(values))).squeeze(-1)
        weights = masked_weights(scores, mask)
        pooled = (values.float() * weights.unsqueeze(-1)).sum(dim=-2).to(values.dtype)
        return pooled, weights


class DinAttention(nn.Module):
    """DIN local activation over concat(q, k, q-k, q*k)."""
    def __init__(self, trainable=True):
        super().__init__()
        if not ENCODER_CONFIG.din_weight_normalization:
            raise NotImplementedError("only normalized masked DIN weights are implemented")
        width = ENCODER_CONFIG.hidden_size
        self.hidden = make_linear(4 * ENCODER_CONFIG.embedding_dim, width, trainable=trainable)
        self.activation = getattr(torch, ENCODER_CONFIG.din_activation)
        self.score = make_linear(width, 1, bias=False, trainable=trainable)

    def forward(self, query, keys, mask):
        q = query.unsqueeze(1).expand_as(keys)
        interaction = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.score(self.activation(self.hidden(interaction))).squeeze(-1)
        weights = masked_weights(scores, mask)
        return (keys.float() * weights.unsqueeze(-1)).sum(dim=1).to(keys.dtype)


class FeatureEmbedding(nn.Embedding):
    def forward(self, indices):
        values = super().forward(indices)
        # Embedding is kept in FP32 for Adam. Cast only the looked-up values
        # to limit memory for long histories under autocast.
        device_type = values.device.type
        if torch.is_autocast_enabled(device_type):
            values = values.to(torch.get_autocast_dtype(device_type))
        return values


class VerifiedWordMatrix(object):
    """A word matrix produced by the strict sidecar loader, with provenance attached."""
    def __init__(self, values, manifest):
        self.values = values
        self.manifest = manifest


class TitleAbstractMetadataEncoder(nn.Module):
    """Active no-body variant: exactly 15 [B,100] field vectors."""
    FIELD_ORDER = tuple(ENCODER_CONFIG.field_order)

    def __init__(self, vocab_sizes, word_matrix=None, entity_matrix=None, type_matrix=None,
                 initializer_seed=None, initialized_test_only=False,
                 word_matrix_verified=False):
        super().__init__()
        d = ENCODER_CONFIG.embedding_dim
        self.initialized_test_only = bool(initialized_test_only)
        if word_matrix is not None and not isinstance(word_matrix, VerifiedWordMatrix):
            raise ValueError("word_matrix requires a VerifiedWordMatrix loaded from a provenance sidecar")
        if word_matrix_verified:
            raise ValueError("word_matrix_verified boolean is deprecated; load a VerifiedWordMatrix sidecar")
        if word_matrix is not None and self.initialized_test_only:
            raise ValueError("initialized_test_only is reserved for synthetic fixtures without pretrained word weights")
        fixture_init = ENCODER_CONFIG.fixture_embedding_initializer
        generator = None if initializer_seed is None else torch.Generator().manual_seed(initializer_seed)

        def emb(field, matrix=None):
            trainable_key = {"cat": "category", "sub_cat": "subcategory"}.get(field, field)
            trainable = TRAINING_CONFIG["embedding_trainable"][trainable_key]
            rows = int(vocab_sizes[field])
            if isinstance(matrix, VerifiedWordMatrix):
                matrix = matrix.values
            if matrix is not None:
                if len(matrix.shape) != 2 or int(matrix.shape[1]) != d:
                    raise ValueError("{} pretrained matrix must have width {}".format(field, d))
                if int(matrix.shape[0]) != rows:
                    raise ValueError("{} pretrained row count does not match the supplied vocabulary".format(field))
                # Copy the read-only memmap before training can mutate its weights.
                weights = torch.tensor(matrix, dtype=torch.float32)
            else:
                weights = torch.empty(rows, d)
                if generator is None:
                    nn.init.xavier_uniform_(weights)
                else:
                    nn.init.uniform_(weights, fixture_init["minval"], fixture_init["maxval"],
                                     generator=generator)
            return FeatureEmbedding.from_pretrained(weights, freeze=not trainable, sparse=True)

        self.uid_emb = emb("uid")
        self.did_emb = emb("did")
        self.cat_emb = emb("cat")
        self.subcat_emb = emb("sub_cat")
        self.entity_emb = emb("entity", entity_matrix)
        self.type_emb = emb("entity_type", type_matrix)
        self.word_emb = emb("word", word_matrix)
        trainable = TRAINING_CONFIG["embedding_trainable"]["attention_and_din"]
        self.text_pool = nn.ModuleDict({
            name: MaskedAttention(trainable=trainable) for name in ("title", "abstract")
        })
        self.entity_pool = MaskedAttention(trainable=trainable)
        self.his_id_pool = MaskedAttention(trainable=trainable)
        self.his_cat_pool = MaskedAttention(trainable=trainable)
        self.text_din = nn.ModuleDict({
            name: DinAttention(trainable=trainable) for name in ("title", "abstract")
        })
        self.entity_din = DinAttention(trainable=trainable)
        self._has_pretrained_words = isinstance(word_matrix, VerifiedWordMatrix)
        self._has_pretrained_entity_ids = entity_matrix is not None
        self._has_pretrained_entity_types = type_matrix is not None

    def _text(self, ids, name):
        x = self.word_emb(ids)
        mask = ids.ne(ENCODER_CONFIG.pad_id)
        return self.text_pool[name](x, mask)[0]

    def _entity_vectors(self, ids, types):
        type_values = self.type_emb(types)
        # The source adds entity and type embeddings before pooling. The slot
        # mask is based on entity IDs; a PAD type on a real entity is not
        # separately zeroed, matching the visible source addition call-site.
        return self.entity_emb(ids) + type_values

    def _entity(self, ids, types):
        vec = self._entity_vectors(ids, types)
        return self.entity_pool(vec, ids.ne(ENCODER_CONFIG.pad_id))[0]

    def _text_history(self, ids, mask, name, query):
        batch, history, length = ids.shape
        flat = ids.reshape(batch * history, length)
        vec = self._text(flat, name).reshape(batch, history, ENCODER_CONFIG.embedding_dim)
        return self.text_din[name](query, vec, mask)

    def _entity_history(self, ids, types, query):
        vec = self._entity_vectors(ids, types)
        return self.entity_din(query, vec, ids.ne(ENCODER_CONFIG.pad_id))

    def call_structured(self, f):
        raw_uid = f["uid"]
        uid_ids = raw_uid.masked_fill(raw_uid >= self.uid_emb.num_embeddings,
                                      ENCODER_CONFIG.unk_id)
        uid = self.uid_emb(uid_ids)
        did = self.did_emb(f["did"])
        his_id_values = self.did_emb(f["his_id"])
        # his_pooling=att is query-free in this reimplementation; upstream passes
        # candidate demb at call-site, but the dependency's attention signature
        # and query use were not recovered.
        his_id = self.his_id_pool(
            his_id_values, f["his_id"].ne(ENCODER_CONFIG.pad_id))[0]
        cat = self.cat_emb(f["cat"])
        subcat = self.subcat_emb(f["sub_cat"])
        title_ent = self._entity(f["title_entities"], f["title_entity_types"])
        abstract_ent = self._entity(f["abstract_entities"], f["abstract_entity_types"])
        his_title_ent = self._entity_history(f["his_title_merge_entities"],
                    f["his_title_merge_entity_types"], title_ent)
        his_abstract_ent = self._entity_history(f["his_abstract_merge_entities"],
                    f["his_abstract_merge_entity_types"], abstract_ent)
        his_cats = self.his_cat_pool(self.cat_emb(f["his_cats"]),
                                     f["his_cats"].ne(ENCODER_CONFIG.pad_id))[0]
        his_subcats = self.his_cat_pool(self.subcat_emb(f["his_sub_cats"]),
                                        f["his_sub_cats"].ne(ENCODER_CONFIG.pad_id))[0]
        return {"uid": uid, "did": did, "his_id": his_id, "cat": cat,
                "sub_cat": subcat, "title_entities": title_ent,
                "abstract_entities": abstract_ent,
                "his_title_merge_entities": his_title_ent,
                "his_abstract_merge_entities": his_abstract_ent,
                "his_cats": his_cats, "his_sub_cats": his_subcats}

    def forward(self, f):
        if not self._has_pretrained_words and not self.initialized_test_only:
            raise ValueError("15-field forward requires a verified word_matrix; pass initialized_test_only=True only for synthetic smoke fixtures")
        structured = self.call_structured(f)
        cur_title = self._text(f["cur_title_tokens"], "title")
        his_title = self._text_history(f["his_title_tokens"],
                    f["his_title_tokens"].ne(ENCODER_CONFIG.pad_id).any(dim=-1),
                    "title", cur_title)
        cur_abstract = self._text(f["cur_abstract_tokens"], "abstract")
        his_abstract = self._text_history(f["his_abstract_tokens"],
                    f["his_abstract_tokens"].ne(ENCODER_CONFIG.pad_id).any(dim=-1),
                    "abstract", cur_abstract)
        outputs = [structured[k] for k in self.FIELD_ORDER[:11]]
        outputs += [cur_title, his_title, cur_abstract, his_abstract]
        result = dict(zip(self.FIELD_ORDER, outputs))
        if tuple(result.keys()) != self.FIELD_ORDER:
            raise AssertionError("15-field output order differs from encoder contract")
        return result


def load_pretrained_matrix(path, expected_rows, expected_width, expected_vocab_sha256, actual_vocab_sha256):
    """Strict .npy loader; caller must prove its row mapping matches this vocab."""
    import numpy as np
    if expected_vocab_sha256 != actual_vocab_sha256:
        raise ValueError("pretrained matrix vocabulary checksum mismatch")
    matrix = np.load(path, mmap_mode="r")
    if matrix.shape != (int(expected_rows), int(expected_width)):
        raise ValueError("pretrained matrix shape {}, expected {}x{}".format(
            matrix.shape, expected_rows, expected_width))
    if not np.isfinite(matrix).all():
        raise ValueError("pretrained matrix contains non-finite values")
    return matrix


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_glove_matrix(path, word_vocab_tsv, expected_vocab_sha256, expected_rows, width=None):
    """Deprecated loader: a vocab hash alone is not provenance for a trained matrix."""
    raise ValueError("use load_glove_artifact with a checksum-verified provenance sidecar")


def load_glove_artifact(matrix_path, manifest_path, word_vocab_tsv, expected_rows, width=None):
    """Load the generated MIND GloVe matrix only with a matching provenance sidecar."""
    import json
    width = ENCODER_CONFIG.embedding_dim if width is None else int(width)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    accepted_statuses = {
        "MIND_TITLE_ABSTRACT_GLOVE_REIMPLEMENTATION_NOT_ORIGINAL_AUTHOR_ARTIFACT",
        "MIND_TITLE_ABSTRACT_GLOVE_REIMPLEMENTATION_TRAINED_ROWS_L2_NORMALIZED",
    }
    if manifest.get("status") not in accepted_statuses:
        raise ValueError("GloVe manifest status is not an accepted trained MIND metadata artifact")
    actual_vocab_hash = sha256_file(word_vocab_tsv)
    if manifest.get("mapping_sha256") != actual_vocab_hash:
        raise ValueError("GloVe sidecar mapping checksum does not match current word vocabulary")
    actual_matrix_hash = sha256_file(matrix_path)
    if manifest.get("matrix_sha256") != actual_matrix_hash:
        raise ValueError("GloVe matrix checksum does not match its provenance sidecar")
    base = Path(__file__).resolve().parents[1]
    def resolve_file(value):
        path = Path(value)
        return path if path.is_file() else base / path
    training_path = resolve_file(manifest.get("training_source_manifest", ""))
    corpus_path = resolve_file(manifest.get("corpus_manifest", ""))
    if not training_path.is_file() or sha256_file(str(training_path)) != manifest.get("training_source_manifest_sha256"):
        raise ValueError("GloVe training provenance manifest is missing or has changed")
    if not corpus_path.is_file() or sha256_file(str(corpus_path)) != manifest.get("corpus_manifest_sha256"):
        raise ValueError("GloVe corpus provenance manifest is missing or has changed")
    with open(training_path, encoding="utf-8") as f:
        training = json.load(f)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    if training.get("status") != "TRAINED_REIMPLEMENTATION_GLOVE":
        raise ValueError("GloVe training manifest is not a completed full run")
    if int(training.get("parameters", {}).get("vector_size", -1)) != int(width):
        raise ValueError("GloVe training manifest vector width mismatch")
    if training.get("files", {}).get("corpus", {}).get("sha256") != corpus.get("corpus_sha256"):
        raise ValueError("GloVe training and corpus manifests refer to different corpus bytes")
    if corpus.get("corpus_sha256") != manifest.get("corpus_sha256"):
        raise ValueError("GloVe sidecar corpus checksum mismatch")
    if training.get("files", {}).get("vectors_text", {}).get("sha256") != manifest.get("trained_vectors_text_sha256"):
        raise ValueError("trained vector text checksum mismatch")
    if manifest.get("status") == "MIND_TITLE_ABSTRACT_GLOVE_REIMPLEMENTATION_TRAINED_ROWS_L2_NORMALIZED":
        original_matrix = resolve_file(manifest.get("original_matrix_path", ""))
        original_manifest = resolve_file(manifest.get("original_matrix_manifest_path", ""))
        if not original_matrix.is_file() or sha256_file(str(original_matrix)) != manifest.get("original_matrix_sha256"):
            raise ValueError("pre-normalized source matrix is missing or has changed")
        if not original_manifest.is_file() or sha256_file(str(original_manifest)) != manifest.get("original_matrix_manifest_sha256"):
            raise ValueError("pre-normalized source sidecar is missing or has changed")
        if manifest.get("fallback_rows", {}).get("normalization_applied") is not False:
            raise ValueError("normalized matrix provenance does not preserve fallback rows")
    matrix = load_pretrained_matrix(matrix_path, expected_rows, width,
                                    actual_vocab_hash, actual_vocab_hash)
    if list(matrix.shape) != manifest.get("matrix_shape"):
        raise ValueError("GloVe sidecar matrix shape differs from loaded array")
    manifest["training_manifest_verified"] = True
    manifest["corpus_manifest_verified"] = True
    return VerifiedWordMatrix(matrix, manifest), manifest
