import pytest
import yaml

from covid_xray.config import CLASS_NAMES, VARIANTS, RunConfig, load_config


def test_class_names_are_canonical_and_ordered():
    assert CLASS_NAMES == ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"name": "run1_raw", "variant": "raw", "model": "densenet121"}))

    cfg = load_config(path)

    assert cfg.name == "run1_raw"
    assert cfg.variant == "raw"
    assert cfg.model == "densenet121"


def test_load_config_applies_spec_defaults(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"name": "r", "variant": "raw", "model": "densenet121"}))

    cfg = load_config(path)

    assert cfg.image_size == 224
    assert cfg.batch_size == 32
    assert cfg.seed == 42
    assert cfg.stage_a_lr == 1e-3
    assert cfg.stage_b_lr == 1e-5
    assert cfg.stage_a_patience == 3
    assert cfg.stage_b_patience == 5
    assert cfg.unfreeze_from == "conv5_block1"


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="variant"):
        RunConfig(name="bad", variant="greyscale", model="densenet121")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="model"):
        RunConfig(name="bad", variant="raw", model="resnet50")


def test_config_is_immutable():
    cfg = RunConfig(name="r", variant="raw", model="densenet121")
    with pytest.raises(Exception):
        cfg.seed = 7


def test_all_four_variants_are_declared():
    assert VARIANTS == ("raw", "masked", "lungs_removed", "downsample8")
