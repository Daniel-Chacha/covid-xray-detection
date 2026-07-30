"""Model builders.

DenseNet121 for runs 1, 2 and 4; a multinomial logistic regression for the
8x8 shortcut probe (run 3).

DenseNet121 was chosen over a from-scratch CNN and over ResNet for the usual
medical-imaging reason — dense feature reuse works well on the low-contrast,
texture-heavy structure of chest radiographs — but the specific architecture
is not the point of this project and is not ablated.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from covid_xray.config import RunConfig

# DenseNet121's final activation, shape (7, 7, 1024) at 224x224 input.
# This is the Grad-CAM target layer. Exact-match lookup: many layers contain
# "relu" in their name, but only the final activation is named exactly "relu".
FEATURE_LAYER = "relu"


def build_densenet121(cfg: RunConfig) -> keras.Model:
    """ImageNet-pretrained DenseNet121 with a fresh 4-class head.

    The base is invoked with `training=False`, which pins every BatchNorm layer
    to inference mode for the lifetime of the model — including Stage B, when
    the convolutional blocks become trainable.
    """
    inputs = keras.Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")

    base = keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(cfg.image_size, cfg.image_size, 3)
    )
    base.trainable = False  # Stage A: head only

    features = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(features)
    x = keras.layers.Dropout(cfg.dropout, name="dropout")(x)
    # float32 output regardless of the mixed-precision policy: softmax in
    # float16 loses precision in the tails, which corrupts the calibration
    # analysis in Task 10.
    outputs = keras.layers.Dense(
        cfg.num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)

    return keras.Model(inputs, outputs, name=f"{cfg.name}_densenet121")


def get_base_model(model: keras.Model) -> keras.Model:
    """The nested DenseNet121 inside a classifier built by `build_densenet121`.

    Found by type rather than by name: Keras version changes have moved model
    naming around, and a `get_layer("densenet121")` lookup that silently starts
    failing would be diagnosed as a training bug rather than a lookup bug.
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError(f"no nested base model found in {model.name!r}")


def set_finetune_trainable(model: keras.Model, unfreeze_from: str) -> int:
    """Stage B: unfreeze the base from `unfreeze_from` onward, BatchNorm excepted.

    Returns the number of layers made trainable.
    """
    base = get_base_model(model)
    base.trainable = True

    names = [layer.name for layer in base.layers]
    matches = [i for i, name in enumerate(names) if name.startswith(unfreeze_from)]
    if not matches:
        raise ValueError(f"no layer starting with {unfreeze_from!r} in {base.name}")
    cutoff = matches[0]

    unfrozen = 0
    for index, layer in enumerate(base.layers):
        if index < cutoff or isinstance(layer, keras.layers.BatchNormalization):
            # BatchNorm is frozen everywhere. `training=False` at the call site
            # already stops statistics updates; freezing here also pins gamma
            # and beta, so the normalisation is fully fixed.
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen += 1

    return unfrozen


def split_feature_and_head(model: keras.Model) -> tuple[keras.Model, keras.Model]:
    """Split into (image -> feature map) and (feature map -> probabilities).

    Grad-CAM needs gradients of the class score with respect to the last
    convolutional feature map. Because the base is a *nested* model, its
    internal tensors are not reachable from the outer graph, so the usual
    single-model Grad-CAM recipe fails. Splitting sidesteps that.
    """
    base = get_base_model(model)
    feature_model = keras.Model(base.inputs, base.get_layer(FEATURE_LAYER).output, name="features")

    feature_shape = feature_model.output.shape[1:]
    head_input = keras.Input(shape=feature_shape, name="features_in")

    x = head_input
    base_position = model.layers.index(base)
    for layer in model.layers[base_position + 1 :]:
        x = layer(x)

    return feature_model, keras.Model(head_input, x, name="head")


def build_probe(seed: int = 42) -> Pipeline:
    """Multinomial logistic regression for the 8x8 shortcut probe.

    Deliberately the weakest plausible model. If *this* separates the classes,
    the label is recoverable from global intensity structure alone and no
    amount of DenseNet accuracy means what it appears to mean.
    """
    # No `multi_class` argument: it was deprecated in scikit-learn 1.5 and
    # REMOVED in 1.7. With lbfgs, multinomial is the behaviour anyway.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
