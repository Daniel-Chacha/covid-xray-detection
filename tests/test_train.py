from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from tensorflow import keras

from covid_xray.config import RunConfig, load_config
from covid_xray.models import get_base_model
from covid_xray.train import build_callbacks, set_global_seed, train_two_stage

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
RUN_NAMES = ("run1_raw", "run2_masked", "run3_probe8", "run4_lungs_removed")


@pytest.fixture
def tiny_cfg():
    return RunConfig(
        name="tiny",
        variant="raw",
        model="densenet121",
        image_size=32,
        batch_size=4,
        stage_a_epochs=1,
        stage_b_epochs=1,
        mixed_precision=False,
    )


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 32, 32, 3)).astype(np.float32)
    y = tf.one_hot(rng.integers(0, 4, size=16), 4).numpy()
    w = np.ones(16, dtype=np.float32)
    train = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(4)
    val = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    return train, val


def test_all_four_configs_load():
    for name in RUN_NAMES:
        assert load_config(CONFIG_DIR / f"{name}.yaml").name == name


def test_configs_cover_every_variant():
    variants = {load_config(CONFIG_DIR / f"{n}.yaml").variant for n in RUN_NAMES}
    assert variants == {"raw", "masked", "downsample8", "lungs_removed"}


def test_callbacks_monitor_validation_macro_f1(tiny_cfg, tmp_path):
    callbacks = build_callbacks(tiny_cfg, tmp_path, stage="a")
    monitored = [c for c in callbacks if hasattr(c, "monitor")]
    assert monitored
    for callback in monitored:
        assert callback.monitor == "val_macro_f1"
        assert callback.mode == "max"


def test_callbacks_include_backup_for_session_death(tiny_cfg, tmp_path):
    names = [type(c).__name__ for c in build_callbacks(tiny_cfg, tmp_path, stage="a")]
    assert "BackupAndRestore" in names
    assert "ModelCheckpoint" in names
    assert "EarlyStopping" in names
    assert "CSVLogger" in names


def test_early_stopping_patience_matches_the_stage(tiny_cfg, tmp_path):
    stop_a = [c for c in build_callbacks(tiny_cfg, tmp_path, "a")
              if isinstance(c, keras.callbacks.EarlyStopping)][0]
    stop_b = [c for c in build_callbacks(tiny_cfg, tmp_path, "b")
              if isinstance(c, keras.callbacks.EarlyStopping)][0]
    assert stop_a.patience == tiny_cfg.stage_a_patience
    assert stop_b.patience == tiny_cfg.stage_b_patience
    assert stop_a.restore_best_weights and stop_b.restore_best_weights


def test_two_stage_training_returns_both_histories(tiny_cfg, tiny_data, tmp_path):
    train_ds, val_ds = tiny_data
    model, history = train_two_stage(tiny_cfg, train_ds, val_ds, tmp_path)

    assert set(history) == {"stage_a", "stage_b"}
    assert "val_macro_f1" in history["stage_a"]
    assert "val_macro_f1" in history["stage_b"]


def test_base_is_trainable_after_stage_b(tiny_cfg, tiny_data, tmp_path):
    train_ds, val_ds = tiny_data
    model, _ = train_two_stage(tiny_cfg, train_ds, val_ds, tmp_path)
    assert get_base_model(model).trainable is True


def test_set_global_seed_makes_initialisation_reproducible():
    set_global_seed(42)
    first = tf.random.uniform((5,)).numpy()
    set_global_seed(42)
    second = tf.random.uniform((5,)).numpy()
    assert np.allclose(first, second)
