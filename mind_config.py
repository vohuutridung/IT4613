"""Single configuration loader shared by preprocessing and model stages.

Values live in ``configs/mind_no_body_15field.json`` so preprocessing,
representations, interaction, and training cannot silently drift apart.
"""
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "mind_no_body_15field.json"
with CONFIG_PATH.open(encoding="utf-8") as _stream:
    CONFIG_DATA = json.load(_stream)


class ConfigSection:
    """Small read-only-by-convention mapping with legacy attribute access."""

    def __init__(self, values):
        self._values = dict(values)

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_dict(self):
        return dict(self._values)


_shared = dict(CONFIG_DATA["shared_architecture"])
_source = CONFIG_DATA["source"]

# Stage 01 keeps the historical body-v17 database contract.  The active
# encoder is a separate no-body profile assembled from the same shared values.
# Keeping these views explicit prevents the model-only variant from changing
# the fingerprint/meaning of the already prepared database.
_preprocessing_values = {
    "source_commit": _source["commit"],
    **_shared,
    **CONFIG_DATA["preprocessing"],
}
_encoder_values = {
    **_shared,
    **CONFIG_DATA["encoder"],
}

PREPROCESSING_CONFIG = ConfigSection(_preprocessing_values)
ENCODER_CONFIG = ConfigSection(_encoder_values)
TRAINING_CONFIG = dict(CONFIG_DATA["training"])
TRAINING_CONFIG["field_order"] = list(ENCODER_CONFIG.field_order)
EXECUTION_CONFIG = ConfigSection(CONFIG_DATA["execution"])
ARTIFACT_CONFIG = ConfigSection({
    "initialization_seed": CONFIG_DATA["artifact_initialization_seed"]
})

if len(ENCODER_CONFIG.field_order) != 15 or any(
        key in ENCODER_CONFIG.field_order for key in ("cur_body", "his_body")):
    raise ValueError("canonical config must describe exactly the approved 15-field no-body variant")
