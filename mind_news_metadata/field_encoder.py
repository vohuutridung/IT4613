"""Independent TensorFlow/Keras field representation encoder; no interaction or head."""
from pathlib import Path

import tensorflow as tf

from .encoder_config import ENCODER_CONFIG
from mind_config import TRAINING_CONFIG


class MaskedAttention(tf.keras.layers.Layer):
    def __init__(self, name=None, trainable=True):
        super(MaskedAttention, self).__init__(name=name, trainable=trainable)
        self.proj = tf.keras.layers.Dense(
            ENCODER_CONFIG.attention_hidden,
            activation=ENCODER_CONFIG.attention_activation,
            name="attention_hidden",
            kernel_initializer=TRAINING_CONFIG["dense_kernel_initializer"],
            bias_initializer=TRAINING_CONFIG["dense_bias_initializer"])
        self.score = tf.keras.layers.Dense(
            1, use_bias=False, name="attention_score",
            kernel_initializer=TRAINING_CONFIG["dense_kernel_initializer"])

    def call(self, values, mask):
        scores = tf.squeeze(self.score(self.proj(values)), axis=-1)
        mask = tf.cast(mask, tf.bool)
        floor = tf.ones_like(scores) * tf.cast(ENCODER_CONFIG.masked_score_floor, scores.dtype)
        scores = tf.where(mask, scores, floor)
        weights = tf.nn.softmax(scores, axis=-1)
        weights = weights * tf.cast(mask, weights.dtype)
        denom = tf.reduce_sum(weights, axis=-1, keepdims=True)
        weights = tf.math.divide_no_nan(weights, denom)
        pooled = tf.reduce_sum(values * tf.expand_dims(weights, -1), axis=-2)
        return pooled, weights


class DinAttention(tf.keras.layers.Layer):
    """DIN local activation: concat(q,k,q-k,q*k), sigmoid hidden, softmax."""
    def __init__(self, name=None, trainable=True):
        if not ENCODER_CONFIG.din_weight_normalization:
            raise NotImplementedError("only normalized masked DIN weights are implemented")
        super(DinAttention, self).__init__(name=name, trainable=trainable)
        self.hidden = tf.keras.layers.Dense(ENCODER_CONFIG.hidden_size,
                                            activation=ENCODER_CONFIG.din_activation,
                                            name="din_hidden",
                                            kernel_initializer=TRAINING_CONFIG["dense_kernel_initializer"],
                                            bias_initializer=TRAINING_CONFIG["dense_bias_initializer"],
                                            trainable=trainable)
        self.score = tf.keras.layers.Dense(1, use_bias=False, name="din_score",
                                           kernel_initializer=TRAINING_CONFIG["dense_kernel_initializer"],
                                           trainable=trainable)
        self.last_query = None
        self.last_weights = None

    def call(self, query, keys, mask):
        q = tf.expand_dims(query, axis=1)
        q = tf.ones_like(keys) * q
        interaction = tf.concat([q, keys, q - keys, q * keys], axis=-1)
        scores = tf.squeeze(self.score(self.hidden(interaction)), axis=-1)
        mask = tf.cast(mask, tf.bool)
        scores = tf.where(
            mask, scores,
            tf.ones_like(scores) * tf.cast(ENCODER_CONFIG.masked_score_floor, scores.dtype))
        weights = tf.nn.softmax(scores, axis=-1)
        weights = weights * tf.cast(mask, weights.dtype)
        weights = tf.math.divide_no_nan(weights, tf.reduce_sum(weights, axis=-1, keepdims=True))
        result = tf.reduce_sum(keys * tf.expand_dims(weights, -1), axis=1)
        self.last_query, self.last_weights = query, weights
        return result


class VerifiedWordMatrix(object):
    """A word matrix produced by the strict sidecar loader, with provenance attached."""
    def __init__(self, values, manifest):
        self.values = values
        self.manifest = manifest


class TitleAbstractMetadataEncoder(tf.keras.Model):
    """Active no-body variant: exactly 15 [B,100] field vectors."""
    FIELD_ORDER = tuple(ENCODER_CONFIG.field_order)

    def __init__(self, vocab_sizes, word_matrix=None, entity_matrix=None, type_matrix=None,
                 initializer_seed=None, initialized_test_only=False,
                 word_matrix_verified=False):
        super(TitleAbstractMetadataEncoder, self).__init__(name=ENCODER_CONFIG.variant)
        d = ENCODER_CONFIG.embedding_dim
        self.initialized_test_only = bool(initialized_test_only)
        if word_matrix is not None and not isinstance(word_matrix, VerifiedWordMatrix):
            raise ValueError("word_matrix requires a VerifiedWordMatrix loaded from a provenance sidecar")
        if word_matrix_verified:
            raise ValueError("word_matrix_verified boolean is deprecated; load a VerifiedWordMatrix sidecar")
        if word_matrix is not None and self.initialized_test_only:
            raise ValueError("initialized_test_only is reserved for synthetic fixtures without pretrained word weights")
        fixture_init = ENCODER_CONFIG.fixture_embedding_initializer
        if initializer_seed is not None:
            init = tf.keras.initializers.RandomUniform(
                minval=fixture_init["minval"], maxval=fixture_init["maxval"],
                seed=int(initializer_seed))
        def emb(field, matrix=None):
            trainable_key = {"cat": "category", "sub_cat": "subcategory"}.get(field, field)
            trainable = TRAINING_CONFIG["embedding_trainable"][trainable_key]
            if matrix is not None:
                if isinstance(matrix, VerifiedWordMatrix):
                    matrix = matrix.values
                matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
                if matrix.shape.ndims != 2 or int(matrix.shape[1]) != d:
                    raise ValueError("{} pretrained matrix must have width {}".format(field, d))
                if int(matrix.shape[0]) != int(vocab_sizes[field]):
                    raise ValueError("{} pretrained row count does not match the supplied vocabulary".format(field))
                return tf.keras.layers.Embedding(matrix.shape[0], int(matrix.shape[1]),
                    embeddings_initializer=tf.keras.initializers.Constant(matrix.numpy()),
                    name=field + "_embedding", trainable=trainable)
            return tf.keras.layers.Embedding(int(vocab_sizes[field]), d,
                    embeddings_initializer=ENCODER_CONFIG.embedding_initializer if initializer_seed is None else init,
                    name=field + "_embedding", trainable=trainable)
        self.uid_emb = emb("uid")
        self.did_emb = emb("did")
        self.cat_emb = emb("cat")
        self.subcat_emb = emb("sub_cat")
        self.entity_emb = emb("entity", entity_matrix)
        self.type_emb = emb("entity_type", type_matrix)
        self.word_emb = emb("word", word_matrix)
        self.text_pool = {name: MaskedAttention(name=name + "_pool",
                                trainable=TRAINING_CONFIG["embedding_trainable"]["attention_and_din"])
                          for name in ("title", "abstract")}
        self.entity_pool = MaskedAttention(name="entity_pool",
                                           trainable=TRAINING_CONFIG["embedding_trainable"]["attention_and_din"])
        self.his_id_pool = MaskedAttention(name="his_id_pool",
                                           trainable=TRAINING_CONFIG["embedding_trainable"]["attention_and_din"])
        self.his_cat_pool = MaskedAttention(name="his_cat_pool",
                                            trainable=TRAINING_CONFIG["embedding_trainable"]["attention_and_din"])
        self.text_din = {name: DinAttention(
                            name="his_" + name + "_din",
                            trainable=TRAINING_CONFIG["embedding_trainable"]["attention_and_din"])
                         for name in ("title", "abstract")}
        self.entity_din = DinAttention(
            name="his_entity_din",
            trainable=TRAINING_CONFIG["embedding_trainable"]["attention_and_din"])
        self.last_queries = {}
        self._has_pretrained_words = isinstance(word_matrix, VerifiedWordMatrix)
        self._has_pretrained_entity_ids = entity_matrix is not None
        self._has_pretrained_entity_types = type_matrix is not None

    def _text(self, ids, name):
        x = self.word_emb(ids)
        mask = tf.not_equal(ids, ENCODER_CONFIG.pad_id)
        return self.text_pool[name](x, mask)[0]

    def _entity_vectors(self, ids, types):
        type_values = self.type_emb(types)
        # The source adds entity and type embeddings before pooling. The slot
        # mask is based on entity IDs; a PAD type on a real entity is not
        # separately zeroed, matching the visible source addition call-site.
        return self.entity_emb(ids) + type_values

    def _entity(self, ids, types):
        vec = self._entity_vectors(ids, types)
        return self.entity_pool(vec, tf.not_equal(ids, ENCODER_CONFIG.pad_id))[0]

    def _text_history(self, ids, mask, name, query):
        shape = tf.shape(ids)
        b, t, l = shape[0], shape[1], shape[2]
        flat = tf.reshape(ids, [-1, l])
        vec = self._text(flat, name)
        vec = tf.reshape(vec, [b, t, ENCODER_CONFIG.embedding_dim])
        self.last_queries["his_" + name] = query
        return self.text_din[name](query, vec, mask)

    def _entity_history(self, ids, types, query, branch):
        vec = self._entity_vectors(ids, types)
        self.last_queries[branch] = query
        return self.entity_din(query, vec, tf.not_equal(ids, ENCODER_CONFIG.pad_id))

    def call_structured(self, f):
        raw_uid = f["uid"]
        uid_ids = tf.where(raw_uid < int(self.uid_emb.input_dim), raw_uid,
                           tf.ones_like(raw_uid) * int(ENCODER_CONFIG.unk_id))
        uid = self.uid_emb(uid_ids)
        did = self.did_emb(f["did"])
        his_id_values = self.did_emb(f["his_id"])
        # his_pooling=att is query-free in this reimplementation; upstream passes
        # candidate demb at call-site, but the dependency's attention signature
        # and query use were not recovered.
        his_id = self.his_id_pool(
            his_id_values, tf.not_equal(f["his_id"], ENCODER_CONFIG.pad_id))[0]
        cat = self.cat_emb(f["cat"])
        subcat = self.subcat_emb(f["sub_cat"])
        title_ent = self._entity(f["title_entities"], f["title_entity_types"])
        abstract_ent = self._entity(f["abstract_entities"], f["abstract_entity_types"])
        his_title_ent = self._entity_history(f["his_title_merge_entities"],
                    f["his_title_merge_entity_types"], title_ent, "his_title_merge_entities")
        his_abstract_ent = self._entity_history(f["his_abstract_merge_entities"],
                    f["his_abstract_merge_entity_types"], abstract_ent, "his_abstract_merge_entities")
        his_cats = self.his_cat_pool(self.cat_emb(f["his_cats"]),
                                     tf.not_equal(f["his_cats"], ENCODER_CONFIG.pad_id))[0]
        his_subcats = self.his_cat_pool(self.subcat_emb(f["his_sub_cats"]),
                                        tf.not_equal(f["his_sub_cats"], ENCODER_CONFIG.pad_id))[0]
        return {"uid": uid, "did": did, "his_id": his_id, "cat": cat,
                "sub_cat": subcat, "title_entities": title_ent,
                "abstract_entities": abstract_ent,
                "his_title_merge_entities": his_title_ent,
                "his_abstract_merge_entities": his_abstract_ent,
                "his_cats": his_cats, "his_sub_cats": his_subcats}

    def call(self, f, training=False):
        if not self._has_pretrained_words and not self.initialized_test_only:
            raise ValueError("15-field forward requires a verified word_matrix; pass initialized_test_only=True only for synthetic smoke fixtures")
        structured = self.call_structured(f)
        cur_title = self._text(f["cur_title_tokens"], "title")
        his_title = self._text_history(f["his_title_tokens"],
                    tf.reduce_any(tf.not_equal(f["his_title_tokens"], ENCODER_CONFIG.pad_id), axis=-1),
                    "title", cur_title)
        cur_abstract = self._text(f["cur_abstract_tokens"], "abstract")
        his_abstract = self._text_history(f["his_abstract_tokens"],
                    tf.reduce_any(tf.not_equal(f["his_abstract_tokens"], ENCODER_CONFIG.pad_id), axis=-1),
                    "abstract", cur_abstract)
        outputs = [structured[k] for k in self.FIELD_ORDER[:11]]
        outputs += [cur_title, his_title, cur_abstract, his_abstract]
        result = dict(zip(self.FIELD_ORDER, outputs))
        self.last_queries["his_title"] = cur_title
        self.last_queries["his_abstract"] = cur_abstract
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
