from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deps import ensure_requirements_installed

ensure_requirements_installed()

import numpy as np


def resolve_compute_backend(requested_backend: str) -> tuple[str, str | None, Any, None, dict[str, Any]]:
    value = (requested_backend or "cpu").strip().lower()
    if value == "cpu":
        return "cpu", None, np, None, {"provider": "numpy", "runtime": "cpu"}
    return "cpu", f"Backend '{requested_backend}' wird nicht unterstuetzt. Nutze CPU.", np, None, {
        "provider": "numpy",
        "runtime": "cpu",
    }


def one_hot(labels: Any, num_classes: int = 10) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    encoded = np.zeros((labels_np.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels_np.shape[0]), labels_np] = 1.0
    return encoded


class SimpleMLP:
    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_size: int = 128,
        hidden_layers: int = 1,
        output_size: int = 10,
        seed: int = 42,
        backend: str = "cpu",
    ) -> None:
        self.requested_backend = backend
        self.backend, self.backend_note, self.xp, _unused_runtime, self.backend_info = resolve_compute_backend(backend)

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
            weight_np = rng.normal(0.0, np.sqrt(2.0 / prev_size), (prev_size, next_size)).astype(np.float32)
            bias_np = np.zeros((1, next_size), dtype=np.float32)
            self.weights.append(self.asarray(weight_np, dtype=np.float32))
            self.biases.append(self.asarray(bias_np, dtype=np.float32))

        # Adam Optimizer State (Mittelwerte und Varianzen)
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # Zeitschritt

    def _ensure_2d_rowwise(self, value: Any) -> np.ndarray:
        value_np = np.asarray(value)
        if value_np.ndim == 2:
            return value_np
        if value_np.ndim == 1:
            if self.input_size > 0 and value_np.size % self.input_size == 0:
                return value_np.reshape((max(1, value_np.size // self.input_size), self.input_size))
            return value_np.reshape((1, -1))
        if value_np.ndim == 0:
            return value_np.reshape((1, 1))
        return value_np

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    def _softmax_numpy(self, x_np: np.ndarray) -> np.ndarray:
        if x_np.ndim == 1:
            x_np = x_np.reshape((1, -1))
        elif x_np.ndim == 0:
            x_np = x_np.reshape((1, 1))
        shifted = x_np - np.max(x_np, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _softmax(self, x: Any) -> np.ndarray:
        return self._softmax_numpy(np.asarray(x, dtype=np.float32))

    def forward(self, x: Any) -> tuple[np.ndarray, dict[str, Any]]:
        x = self.asarray(x, dtype=np.float32)
        x = self._ensure_2d_rowwise(x)
        activations: list[np.ndarray] = [x]
        pre_activations: list[np.ndarray] = []

        a = x
        for layer_idx in range(len(self.weights) - 1):
            z = (a @ self.weights[layer_idx]) + self.biases[layer_idx]
            a = self._relu(z)
            pre_activations.append(z)
            activations.append(a)

        z_out = (a @ self.weights[-1]) + self.biases[-1]
        probs = self._softmax(z_out)
        pre_activations.append(z_out)
        activations.append(probs)

        cache = {"activations": activations, "pre_activations": pre_activations, "probs": probs}
        return probs, cache

    def cross_entropy_loss(self, probs: Any, y_one_hot: Any) -> float:
        probs_np = self.to_numpy(probs)
        labels_np = self.to_numpy(y_one_hot)
        if probs_np.ndim == 1:
            probs_np = probs_np.reshape((1, -1))
        if labels_np.ndim == 1:
            labels_np = labels_np.reshape((1, -1))
        eps = 1e-12
        losses = -np.sum(labels_np * np.log(probs_np + eps), axis=1)
        return float(np.mean(losses))

    def train_batch(self, x_batch: Any, y_batch_one_hot: Any, learning_rate: float) -> tuple[float, float]:
        x_batch = self.asarray(x_batch, dtype=np.float32)
        y_batch_one_hot = self.asarray(y_batch_one_hot, dtype=np.float32)
        x_batch = self._ensure_2d_rowwise(x_batch)
        y_batch_one_hot = self._ensure_2d_rowwise(y_batch_one_hot)

        probs, cache = self.forward(x_batch)

        batch_size = int(x_batch.shape[0])
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]

        grads_w: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        grads_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]

        dz = (probs - y_batch_one_hot) / float(batch_size)
        last_idx = len(self.weights) - 1
        grads_w[last_idx] = activations[last_idx].T @ dz
        grads_b[last_idx] = np.sum(dz, axis=0, keepdims=True)

        for layer_idx in range(last_idx - 1, -1, -1):
            da = dz @ self.weights[layer_idx + 1].T
            dz = da * self._relu_derivative(pre_activations[layer_idx])
            grads_w[layer_idx] = activations[layer_idx].T @ dz
            grads_b[layer_idx] = np.sum(dz, axis=0, keepdims=True)

        # Adam Hyperparameter
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        self.t += 1

        for layer_idx in range(len(self.weights)):
            # Adam Update fuer Gewichte
            self.m_w[layer_idx] = beta1 * self.m_w[layer_idx] + (1 - beta1) * grads_w[layer_idx]
            self.v_w[layer_idx] = beta2 * self.v_w[layer_idx] + (1 - beta2) * (grads_w[layer_idx] ** 2)
            
            m_w_corr = self.m_w[layer_idx] / (1 - beta1**self.t)
            v_w_corr = self.v_w[layer_idx] / (1 - beta2**self.t)
            
            self.weights[layer_idx] -= learning_rate * m_w_corr / (np.sqrt(v_w_corr) + epsilon)

            # Adam Update fuer Biases
            self.m_b[layer_idx] = beta1 * self.m_b[layer_idx] + (1 - beta1) * grads_b[layer_idx]
            self.v_b[layer_idx] = beta2 * self.v_b[layer_idx] + (1 - beta2) * (grads_b[layer_idx] ** 2)
            
            m_b_corr = self.m_b[layer_idx] / (1 - beta1**self.t)
            v_b_corr = self.v_b[layer_idx] / (1 - beta2**self.t)
            
            self.biases[layer_idx] -= learning_rate * m_b_corr / (np.sqrt(v_b_corr) + epsilon)

        loss = self.cross_entropy_loss(probs, y_batch_one_hot)
        probs_np = self.to_numpy(probs)
        labels_np = self.to_numpy(y_batch_one_hot)
        predictions = np.argmax(probs_np, axis=1)
        labels = np.argmax(labels_np, axis=1)
        accuracy = float(np.mean(predictions == labels))
        return loss, accuracy

    def one_hot_labels(self, labels: Any, num_classes: int = 10) -> np.ndarray:
        encoded_np = one_hot(self.to_numpy(labels), num_classes=num_classes)
        return self.asarray(encoded_np, dtype=np.float32)

    def predict_proba(self, x: Any) -> np.ndarray:
        probs, _ = self.forward(x)
        return probs

    def predict(self, x: Any) -> np.ndarray:
        probs = self.to_numpy(self.predict_proba(x))
        return np.argmax(probs, axis=1)

    def evaluate(self, x: Any, y: Any) -> tuple[float, float]:
        probs = self.predict_proba(x)
        y_one_hot = self.one_hot_labels(y, self.output_size)
        loss = self.cross_entropy_loss(probs, y_one_hot)

        probs_np = self.to_numpy(probs)
        y_np = self.to_numpy(y)
        acc = float(np.mean(np.argmax(probs_np, axis=1) == y_np))
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
            payload[f"w{idx}"] = self.to_numpy(weight)
            payload[f"b{idx}"] = self.to_numpy(bias)
        np.savez(target, **payload)

    @classmethod
    def load(cls, path: str | Path, backend: str = "cpu") -> tuple["SimpleMLP", dict[str, Any]]:
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
                backend=backend,
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
                model.weights.append(model.asarray(data[w_key].astype(np.float32), dtype=np.float32))
                model.biases.append(model.asarray(data[b_key].astype(np.float32), dtype=np.float32))

            if not model.weights or not model.biases:
                model.weights = [
                    model.asarray(data["w1"].astype(np.float32), dtype=np.float32),
                    model.asarray(data["w2"].astype(np.float32), dtype=np.float32),
                ]
                model.biases = [
                    model.asarray(data["b1"].astype(np.float32), dtype=np.float32),
                    model.asarray(data["b2"].astype(np.float32), dtype=np.float32),
                ]
                model.hidden_layers = 1

            model.layer_sizes = [int(model.weights[0].shape[0])]
            for weight in model.weights:
                model.layer_sizes.append(int(weight.shape[1]))
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

    def asarray(self, value: Any, dtype: Any | None = None) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def _to_float(self, value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
