"""Run configuration.

One YAML file per experiment. All four runs share a single code path and
differ only by `variant` and `model`, which keeps the comparison honest —
no run gets an accidental advantage from a different pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

# Alphabetical, and identical to the dataset's folder names. Fixing the order
# here fixes the label indices everywhere: COVID=0, Lung_Opacity=1,
# Normal=2, Viral Pneumonia=3. Note the space in the last name.
CLASS_NAMES: tuple[str, ...] = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")

# The four input transforms. See spec section 5.
#   raw            — unmodified image (run 1, headline)
#   masked         — everything outside the lung mask zeroed (run 2)
#   lungs_removed  — everything INSIDE the lung mask zeroed (run 4, probe)
#   downsample8    — raw image at 8x8, fed to a linear model (run 3, probe)
VARIANTS: tuple[str, ...] = ("raw", "masked", "lungs_removed", "downsample8")

MODELS: tuple[str, ...] = ("densenet121", "logreg8")

# Environment variable holding the dataset root, so the same manifest CSVs
# work unchanged across machines.
DATA_ROOT_ENV = "COVID_XRAY_DATA_ROOT"

# The audit's control pair: both classes are adult, so the pediatric confound
# is absent and any remaining separability is source artefact or pathology.
CONTROL_PAIR: tuple[str, str] = ("COVID", "Lung_Opacity")


@dataclass(frozen=True)
class RunConfig:
    name: str
    variant: str
    model: str
    image_size: int = 224
    batch_size: int = 32
    seed: int = 42
    dropout: float = 0.3
    stage_a_epochs: int = 8
    stage_a_lr: float = 1e-3
    stage_a_patience: int = 3
    stage_b_epochs: int = 15
    stage_b_lr: float = 1e-5
    stage_b_patience: int = 5
    unfreeze_from: str = "conv5_block1"
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {VARIANTS}")
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; expected one of {MODELS}")
        if self.image_size < 32:
            raise ValueError(f"image_size {self.image_size} is implausibly small")

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    """Load a run config from YAML, applying spec defaults for absent keys."""
    with open(path) as handle:
        payload = yaml.safe_load(handle) or {}
    return RunConfig(**payload)
