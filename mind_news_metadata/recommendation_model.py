"""Stage 03 reimplementation: 15-field interaction and click prediction head.

This module returns candidate logits/probabilities. It contains no candidate
softmax, loss, sampling, or data access.
"""
import tensorflow as tf

from .field_encoder import TitleAbstractMetadataEncoder
from mind_config import ENCODER_CONFIG, TRAINING_CONFIG


# Class-body comprehensions execute in their own scope, so they cannot resolve
# a sibling class attribute such as ``FIELD_COUNT``. Keep the pair definition
# at module scope and expose it on the layer as the public contract.
_FIELD_COUNT = len(ENCODER_CONFIG.field_order)
_FIELD_PAIRS = tuple((i, j) for i in range(_FIELD_COUNT) for j in range(i + 1, _FIELD_COUNT))


class FieldInteraction(tf.keras.layers.Layer):
    """Flatten 15 fields and append one dot product for every i < j pair."""
    FIELD_COUNT = _FIELD_COUNT
    WIDTH = ENCODER_CONFIG.embedding_dim
    PAIRS = _FIELD_PAIRS

    def call(self, fields):
        stacked = tf.stack(fields, axis=1)  # [B, 15, 100], contract order supplied by caller
        flat = tf.reshape(stacked, [tf.shape(stacked)[0], self.FIELD_COUNT * self.WIDTH])
        dots = tf.stack([
            tf.reduce_sum(stacked[:, i, :] * stacked[:, j, :], axis=-1)
            for i, j in self.PAIRS
        ], axis=1)
        return tf.concat([flat, dots], axis=1), stacked, dots


class NoBodyRecommendationModel(tf.keras.Model):
    """The authorized 15-field variant through probability prediction."""
    def __init__(self, field_encoder, name="mind_no_body_15field_recommendation"):
        super().__init__(name=name)
        if TRAINING_CONFIG["mlp_dropout"] != 0.0 or TRAINING_CONFIG["mlp_batch_norm"]:
            raise NotImplementedError("the locked no-body reimplementation uses dropout=0 and batch_norm=false")
        if TRAINING_CONFIG["weight_decay"] != 0.0:
            raise NotImplementedError("weight decay is not implemented by the selected Adam reimplementation")
        self.field_encoder = field_encoder
        self.interaction = FieldInteraction(name="field_concat_dot")
        # The layer widths and activation come from the canonical model config.
        # Keras initializers remain a declared reimplementation choice.
        self.hidden = [
            tf.keras.layers.Dense(width, activation=TRAINING_CONFIG["mlp_activation"],
                                  name="mlp_{}".format(width),
                                  kernel_initializer=TRAINING_CONFIG["dense_kernel_initializer"],
                                  bias_initializer=TRAINING_CONFIG["dense_bias_initializer"],
                                  trainable=TRAINING_CONFIG["embedding_trainable"]["mlp"])
            for width in TRAINING_CONFIG["mlp_widths"]
        ]
        self.logit_layer = tf.keras.layers.Dense(
            1, name="click_logit",
            kernel_initializer=TRAINING_CONFIG["dense_kernel_initializer"],
            bias_initializer=TRAINING_CONFIG["dense_bias_initializer"],
            trainable=TRAINING_CONFIG["embedding_trainable"]["mlp"])

    def call(self, inputs, training=False):
        fields = self.field_encoder(inputs, training=training)
        if tuple(fields.keys()) != self.field_encoder.FIELD_ORDER:
            raise ValueError("field order does not match the locked 15-field contract")
        x, stacked, dots = self.interaction([fields[k] for k in self.field_encoder.FIELD_ORDER])
        interaction_values = x
        for layer in self.hidden:
            x = layer(x, training=training)
        logits = tf.squeeze(self.logit_layer(x), axis=-1)
        return {"logits": logits, "probabilities": tf.math.sigmoid(logits),
                "fields": fields, "stacked_fields": stacked,
                "pairwise_dots": dots, "interaction": interaction_values}


def pointwise_bce_sum(labels, logits):
    """Unreduced sum BCE from logits; caller validates labeled-only rows."""
    labels = tf.cast(labels, tf.float32)
    logits = tf.cast(logits, tf.float32)
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
