"""Two-stage transfer-learning loop.

Stage A trains the new head against a frozen base at lr 1e-3. Stage B unfreezes
the top block at lr 1e-5. Selection is on validation macro-F1 rather than
accuracy, because accuracy is dominated by the Normal class and would happily
select a model that never predicts Viral Pneumonia.

BackupAndRestore is included because Kaggle sessions terminate without warning;
a killed session resumes from the last completed epoch rather than from zero.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from covid_xray.config import RunConfig
from covid_xray.models import build_densenet121, set_finetune_trainable

MONITOR = "val_macro_f1"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def enable_mixed_precision(cfg: RunConfig) -> None:
    """Halves activation memory on a T4 and roughly doubles throughput.

    Safe here because the output layer is pinned to float32 in models.py.
    Called from the notebook, never from train_two_stage — the policy is
    global process state and would otherwise leak between tests.
    """
    if cfg.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")


def _metrics() -> list:
    return [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.F1Score(average="macro", name="macro_f1"),
    ]


def build_callbacks(cfg: RunConfig, output_dir: str | Path, stage: str) -> list:
    """Checkpoint, early-stop, log, and back up. Stage is 'a' or 'b'."""
    output_dir = Path(output_dir)
    (output_dir / "backup").mkdir(parents=True, exist_ok=True)

    patience = cfg.stage_a_patience if stage == "a" else cfg.stage_b_patience

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{cfg.name}_stage{stage}.keras"),
            monitor=MONITOR,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=MONITOR,
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / f"{cfg.name}_stage{stage}.csv")),
        keras.callbacks.BackupAndRestore(backup_dir=str(output_dir / "backup")),
    ]


def train_two_stage(
    cfg: RunConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    output_dir: str | Path,
) -> tuple[keras.Model, dict]:
    """Run Stage A then Stage B. Returns the fitted model and both histories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.seed)

    model = build_densenet121(cfg)

    # --- Stage A: head only -------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.stage_a_lr),
        loss="categorical_crossentropy",
        metrics=_metrics(),
    )
    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.stage_a_epochs,
        callbacks=build_callbacks(cfg, output_dir, stage="a"),
        # Shuffling is the dataset's job (see data.py); saying so silences
        # Keras's warning that it is ignoring this argument.
        shuffle=False,
        verbose=1,
    )

    # --- Stage B: fine-tune the top block ----------------------------------
    unfrozen = set_finetune_trainable(model, cfg.unfreeze_from)
    print(f"stage B: unfroze {unfrozen} layers from {cfg.unfreeze_from!r} (BatchNorm excluded)")

    # Recompile so the optimizer state is fresh and the new trainable set is picked up.
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.stage_b_lr),
        loss="categorical_crossentropy",
        metrics=_metrics(),
    )
    history_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.stage_b_epochs,
        callbacks=build_callbacks(cfg, output_dir, stage="b"),
        shuffle=False,
        verbose=1,
    )


    model.save(output_dir / f"{cfg.name}_final.keras")

    history = {
        "stage_a": {k: [float(v) for v in vs] for k, vs in history_a.history.items()},
        "stage_b": {k: [float(v) for v in vs] for k, vs in history_b.history.items()},
    }
    with open(output_dir / f"{cfg.name}_history.json", "w") as handle:
        json.dump({"config": cfg.to_dict(), "history": history}, handle, indent=2)

    return model, history
