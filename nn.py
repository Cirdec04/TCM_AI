from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deps import ensure_requirements_installed

ensure_requirements_installed()

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    labels = labels.astype(np.int64)
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


class SimpleMLP:
    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_size: int = 128,
        hidden_layers: int = 1,
        output_size: int = 10,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = max(1, int(hidden_layers))
        self.output_size = output_size

        layer_sizes = [self.input_size] + [self.hidden_size] * self.hidden_layers + [self.output_size]
        self.layer_sizes = layer_sizes

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for prev_size, next_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
            self.weights.append(
                rng.normal(0.0, np.sqrt(2.0 / prev_size), (prev_size, next_size)).astype(np.float32)
            )
            self.biases.append(np.zeros((1, next_size), dtype=np.float32))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        activations: list[np.ndarray] = [x]
        pre_activations: list[np.ndarray] = []

        a = x
        for layer_idx in range(len(self.weights) - 1):
            z = a @ self.weights[layer_idx] + self.biases[layer_idx]
            a = self._relu(z)
            pre_activations.append(z)
            activations.append(a)

        z_out = a @ self.weights[-1] + self.biases[-1]
        probs = self._softmax(z_out)
        pre_activations.append(z_out)
        activations.append(probs)

        cache = {"activations": activations, "pre_activations": pre_activations, "probs": probs}
        return probs, cache

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, y_one_hot: np.ndarray) -> float:
        eps = 1e-12
        losses = -np.sum(y_one_hot * np.log(probs + eps), axis=1)
        return float(np.mean(losses))

    def train_batch(self, x_batch: np.ndarray, y_batch_one_hot: np.ndarray, learning_rate: float) -> tuple[float, float]:
        probs, cache = self.forward(x_batch)

        batch_size = x_batch.shape[0]
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]

        grads_w: list[np.ndarray] = [np.empty_like(w) for w in self.weights]
        grads_b: list[np.ndarray] = [np.empty_like(b) for b in self.biases]

        dz = (probs - y_batch_one_hot) / batch_size
        last_idx = len(self.weights) - 1
        grads_w[last_idx] = activations[last_idx].T @ dz
        grads_b[last_idx] = np.sum(dz, axis=0, keepdims=True)

        for layer_idx in range(last_idx - 1, -1, -1):
            da = dz @ self.weights[layer_idx + 1].T
            dz = da * self._relu_derivative(pre_activations[layer_idx])
            grads_w[layer_idx] = activations[layer_idx].T @ dz
            grads_b[layer_idx] = np.sum(dz, axis=0, keepdims=True)

        for layer_idx in range(len(self.weights)):
            self.weights[layer_idx] -= learning_rate * grads_w[layer_idx]
            self.biases[layer_idx] -= learning_rate * grads_b[layer_idx]

        loss = self.cross_entropy_loss(probs, y_batch_one_hot)
        predictions = np.argmax(probs, axis=1)
        labels = np.argmax(y_batch_one_hot, axis=1)
        accuracy = float(np.mean(predictions == labels))
        return loss, accuracy

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(x)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        probs = self.predict_proba(x)
        y_one_hot = one_hot(y, self.output_size)
        loss = self.cross_entropy_loss(probs, y_one_hot)
        acc = float(np.mean(np.argmax(probs, axis=1) == y))
        return loss, acc

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = metadata or {}
        payload: dict[str, Any] = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "layer_sizes": np.array(self.layer_sizes, dtype=np.int32),
            "metadata": json.dumps(metadata),
        }
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=False), start=1):
            payload[f"w{idx}"] = weight
            payload[f"b{idx}"] = bias
        np.savez(target, **payload)

    @classmethod
    def load(cls, path: str | Path) -> tuple["SimpleMLP", dict[str, Any]]:
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            keys = set(data.files)
            input_size = int(data["input_size"])
            hidden_size = int(data["hidden_size"])
            output_size = int(data["output_size"])
            hidden_layers = int(data["hidden_layers"]) if "hidden_layers" in keys else 1

            model = cls(
                input_size=input_size,
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
                output_size=output_size,
            )

            if "layer_sizes" in keys:
                total_layers = len(np.array(data["layer_sizes"]).tolist()) - 1
            else:
                total_layers = hidden_layers + 1

            model.weights = []
            model.biases = []
            for idx in range(1, total_layers + 1):
                w_key = f"w{idx}"
                b_key = f"b{idx}"
                if w_key not in keys or b_key not in keys:
                    continue
                model.weights.append(data[w_key].astype(np.float32))
                model.biases.append(data[b_key].astype(np.float32))

            if not model.weights or not model.biases:
                model.weights = [data["w1"].astype(np.float32), data["w2"].astype(np.float32)]
                model.biases = [data["b1"].astype(np.float32), data["b2"].astype(np.float32)]
                model.hidden_layers = 1

            model.layer_sizes = [model.weights[0].shape[0]]
            for weight in model.weights:
                model.layer_sizes.append(weight.shape[1])
            model.hidden_size = model.layer_sizes[1] if len(model.layer_sizes) > 2 else hidden_size
            model.output_size = model.layer_sizes[-1]
            model.input_size = model.layer_sizes[0]
            model.hidden_layers = len(model.weights) - 1

            metadata_raw = str(data["metadata"])

        metadata: dict[str, Any]
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
        return model, metadata
