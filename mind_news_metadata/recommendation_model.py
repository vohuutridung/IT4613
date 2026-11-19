"""15-field interaction and pointwise click prediction."""
import torch
from torch import nn
from torch.nn import functional as F

from .field_encoder import make_linear
from mind_config import ENCODER_CONFIG, TRAINING_CONFIG


_FIELD_COUNT = len(ENCODER_CONFIG.field_order)
_FIELD_PAIRS = tuple((i, j) for i in range(_FIELD_COUNT) for j in range(i + 1, _FIELD_COUNT))


class FieldInteraction(nn.Module):
    """Flatten the fields and append one dot product for every i < j pair."""

    FIELD_COUNT = _FIELD_COUNT
    WIDTH = ENCODER_CONFIG.embedding_dim
    PAIRS = _FIELD_PAIRS

    def __init__(self):
        super().__init__()
        rows, cols = zip(*self.PAIRS)
        self.register_buffer("rows", torch.tensor(rows), persistent=False)
        self.register_buffer("cols", torch.tensor(cols), persistent=False)

    def forward(self, fields):
        stacked = torch.stack(fields, dim=1).float()
        # Compute interactions in FP32 even when the MLP uses autocast.
        with torch.autocast(stacked.device.type, enabled=False):
            products = torch.bmm(stacked, stacked.transpose(1, 2))
        dots = products[:, self.rows, self.cols]
        return torch.cat([stacked.flatten(1), dots], dim=1), stacked, dots


class NoBodyRecommendationModel(nn.Module):
    def __init__(self, field_encoder):
        super().__init__()
        if TRAINING_CONFIG["mlp_dropout"] != 0.0 or TRAINING_CONFIG["mlp_batch_norm"]:
            raise NotImplementedError("the 15-field model uses dropout=0 and batch_norm=false")
        self.field_encoder = field_encoder
        self.interaction = FieldInteraction()
        trainable = TRAINING_CONFIG["embedding_trainable"]["mlp"]
        input_size = self.interaction.FIELD_COUNT * self.interaction.WIDTH + len(self.interaction.PAIRS)
        widths = [input_size, *TRAINING_CONFIG["mlp_widths"]]
        self.hidden = nn.ModuleList(
            make_linear(in_width, out_width, trainable=trainable)
            for in_width, out_width in zip(widths, widths[1:])
        )
        self.activation = getattr(torch, TRAINING_CONFIG["mlp_activation"])
        self.logit_layer = make_linear(widths[-1], 1, trainable=trainable)

    def score_fields(self, interaction):
        x = interaction
        for layer in self.hidden:
            x = self.activation(layer(x))
        with torch.autocast(x.device.type, enabled=False):
            return self.logit_layer(x.float()).squeeze(-1)

    def forward(self, inputs):
        fields = self.field_encoder(inputs)
        if tuple(fields) != self.field_encoder.FIELD_ORDER:
            raise ValueError("field order does not match the locked 15-field contract")
        interaction, stacked, dots = self.interaction(list(fields.values()))
        logits = self.score_fields(interaction)
        return {"logits": logits, "probabilities": logits.sigmoid(), "fields": fields,
                "stacked_fields": stacked, "pairwise_dots": dots, "interaction": interaction}


def pointwise_bce_sum(labels, logits):
    """Unreduced sum BCE from logits; caller validates labeled-only rows."""
    return F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="sum")
