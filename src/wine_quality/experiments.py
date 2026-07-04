"""Experiment presets — version-controlled, typed, reviewable.

Each entry is a complete TrainConfig; select one with ``experiment=<name>``
and override anything on top::

    python src/wine_quality/train.py experiment=example model.lr=3e-4

Presets are plain Python, so building variants from each other is ordinary
code: ``model_copy(update=...)``, helper functions, loops over widths — no
YAML inheritance rules to memorize.
"""

from wine_quality.configs import DataConfig, ModelConfig, TrainConfig, TrainerConfig

EXPERIMENTS: dict[str, TrainConfig] = {
    # `base` is what runs when no experiment= is given.
    "base": TrainConfig(),
    "example": TrainConfig(
        data=DataConfig(batch_size=128),
        model=ModelConfig(hidden_dim=256, lr=1e-3, weight_decay=1e-4),
        trainer=TrainerConfig(max_epochs=50),
    ),
    # The settings behind the walkthrough's headline number.
    "wine": TrainConfig(
        model=ModelConfig(lr=1e-3),
        trainer=TrainerConfig(max_epochs=30, patience=10),
    ),
}
