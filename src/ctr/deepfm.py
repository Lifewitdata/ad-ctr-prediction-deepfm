"""A DeepFM implementation in TensorFlow / Keras 3 (functional API).

DeepFM (Guo et al., 2017) combines three parts on top of shared categorical
embeddings:
- a linear part (one weight per feature value) for low-order effects,
- a factorisation-machine part modelling all pairwise interactions as dot
  products of embeddings, and
- a deep MLP over the concatenated embeddings for higher-order interactions.

Simplification stated for transparency: the 13 numeric inputs enter as a single
projected field (one Dense layer of dimension k) plus raw values in the MLP,
rather than as 13 individually embedded fields.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras  # noqa: E402
import numpy as np  # noqa: E402

_SEED_SET = False


def set_seed(seed: int) -> None:
    global _SEED_SET
    keras.utils.set_random_seed(seed)
    if not _SEED_SET:
        try:
            import tensorflow as tf

            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - older TF fallbacks
            pass
        _SEED_SET = True


@keras.saving.register_keras_serializable(package="ctr")
class FMInteraction(keras.layers.Layer):
    """Second-order factorisation-machine term for a stack of field embeddings.

    Input shape (batch, n_fields, k); returns (batch, 1) with
    0.5 * sum_j[ (sum_f e_fj)^2 - sum_f e_fj^2 ].
    """

    def call(self, embeddings):
        summed = keras.ops.sum(embeddings, axis=1)
        squared_summed = keras.ops.sum(keras.ops.square(embeddings), axis=1)
        pairwise = 0.5 * (keras.ops.square(summed) - squared_summed)
        return keras.ops.sum(pairwise, axis=-1, keepdims=True)


def build_deepfm(
    numeric_dim: int,
    vocab_sizes: list[int],
    embedding_dim: int = 8,
    hidden_units: tuple[int, ...] = (64, 32),
    dropout: float = 0.1,
    name: str = "deepfm",
) -> keras.Model:
    num_input = keras.Input(shape=(numeric_dim,), name="num", dtype="float32")

    cat_inputs, deep_fields, linear_terms, fm_fields = [], [], [], []
    for i, size in enumerate(vocab_sizes):
        field = f"C{i + 1}"
        inp = keras.Input(shape=(1,), name=field, dtype="int32")
        emb = keras.layers.Embedding(size, embedding_dim, name=f"emb_{field}")(inp)
        lin = keras.layers.Embedding(size, 1, name=f"lin_{field}")(inp)
        cat_inputs.append(inp)
        deep_fields.append(keras.layers.Flatten()(emb))
        linear_terms.append(keras.layers.Flatten()(lin))
        fm_fields.append(emb)

    num_field = keras.layers.Reshape((1, embedding_dim), name="num_field")(  # noqa: E501
        keras.layers.Dense(embedding_dim, use_bias=False, name="num_projection")(num_input)
    )
    fm_input = keras.layers.Concatenate(axis=1, name="fm_fields")(fm_fields + [num_field])
    fm_term = FMInteraction(name="fm_term")(fm_input)

    linear_cat = keras.layers.Add(name="linear_categorical")(linear_terms)
    linear_num = keras.layers.Dense(1, use_bias=False, name="linear_numeric")(num_input)

    deep_input = keras.layers.Concatenate(name="deep_input")(deep_fields + [num_input])
    hidden = keras.layers.Dense(hidden_units[0], activation="relu", name="hidden_1")(deep_input)
    hidden = keras.layers.Dropout(dropout, name="dropout")(hidden)
    hidden = keras.layers.Dense(hidden_units[1], activation="relu", name="hidden_2")(hidden)
    deep_term = keras.layers.Dense(1, name="deep_output")(hidden)

    logit = keras.layers.Add(name="logit")([fm_term, linear_cat, linear_num, deep_term])
    probability = keras.layers.Activation("sigmoid", name="click_probability")(logit)
    return keras.Model(inputs=[num_input] + cat_inputs, outputs=probability, name=name)


class DeepFMWrapper:
    """Training / inference wrapper exposing the same interface as the other models."""

    def __init__(
        self,
        numeric_dim: int,
        vocab_sizes: list[int],
        embedding_dim: int = 8,
        hidden_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.batch_size = batch_size
        self.model_ = build_deepfm(numeric_dim, vocab_sizes, embedding_dim, hidden_units, dropout)
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryCrossentropy(name="logloss"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        self.history_ = None

    def fit(self, X, y, validation_data=None, epochs: int = 8, patience: int = 2) -> "DeepFMWrapper":
        y = np.asarray(y).reshape(-1, 1)  # the model outputs (n, 1) probabilities
        if validation_data is not None:
            x_val, y_val = validation_data
            validation_data = (x_val, np.asarray(y_val).reshape(-1, 1))
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                min_delta=1e-4,
            )
        ]
        self.history_ = self.model_.fit(
            X,
            y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.model_.predict(X, batch_size=self.batch_size, verbose=0).ravel().astype("float64")
        return np.column_stack([1 - p, p])

    def save(self, path) -> None:
        self.model_.save(path)

    @staticmethod
    def load(path) -> keras.Model:
        return keras.models.load_model(path)
