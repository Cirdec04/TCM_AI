from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deps import ensure_requirements_installed

ensure_requirements_installed()

import numpy as np
try:
    # Compute backend option: GPU via CuPy (falls installiert), sonst CPU via NumPy.
    import cupy as cp  # Optional: GPU backend via CuPy
except Exception:  # noqa: BLE001
    cp = None


def resolve_compute_backend(requested_backend: str) -> tuple[str, str | None]:
    # Zentraler Schalter fuer CPU/GPU-Auswahl mit sauberem Fallback.
    value = (requested_backend or "cpu").strip().lower()
    if value not in {"cpu", "gpu"}:
        return "cpu", f"Unbekannter Backend-Wert '{requested_backend}', nutze CPU."
    if value == "gpu" and cp is None:
        return "cpu", "GPU angefragt, aber CuPy ist nicht verfuegbar. Nutze CPU."
    return value, None


def one_hot(labels: Any, num_classes: int = 10, xp: Any = np) -> Any:
    labels = labels.astype(xp.int64)
    encoded = xp.zeros((labels.shape[0], num_classes), dtype=xp.float32)
    encoded[xp.arange(labels.shape[0]), labels] = 1.0
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
        self.backend, self.backend_note = resolve_compute_backend(backend)
        self.xp = cp if self.backend == "gpu" else np

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
                self.xp.asarray(
                    rng.normal(0.0, np.sqrt(2.0 / prev_size), (prev_size, next_size)).astype(np.float32)
                )
            )
            self.biases.append(self.xp.zeros((1, next_size), dtype=self.xp.float32))

    def _relu(self, x: Any) -> Any:
        return self.xp.maximum(0.0, x)

    def _relu_derivative(self, x: Any) -> Any:
        return (x > 0.0).astype(self.xp.float32)

    def _softmax(self, x: Any) -> Any:
        shifted = x - self.xp.max(x, axis=1, keepdims=True)
        exp_values = self.xp.exp(shifted)
        return exp_values / self.xp.sum(exp_values, axis=1, keepdims=True)

    def forward(self, x: Any) -> tuple[Any, dict[str, Any]]:
        x = self.asarray(x, dtype=self.xp.float32)
        activations: list[Any] = [x]
        pre_activations: list[Any] = []

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

    def cross_entropy_loss(self, probs: Any, y_one_hot: Any) -> float:
        eps = 1e-12
        losses = -self.xp.sum(y_one_hot * self.xp.log(probs + eps), axis=1)
        return self._to_float(self.xp.mean(losses))

    def train_batch(self, x_batch: Any, y_batch_one_hot: Any, learning_rate: float) -> tuple[float, float]:
        probs, cache = self.forward(x_batch)

        batch_size = x_batch.shape[0]
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]

        grads_w: list[np.ndarray] = [np.empty_like(w) for w in self.weights]
        grads_b: list[np.ndarray] = [np.empty_like(b) for b in self.biases]

        dz = (probs - y_batch_one_hot) / batch_size
        last_idx = len(self.weights) - 1
        grads_w[last_idx] = activations[last_idx].T @ dz
        grads_b[last_idx] = self.xp.sum(dz, axis=0, keepdims=True)

        for layer_idx in range(last_idx - 1, -1, -1):
            da = dz @ self.weights[layer_idx + 1].T
            dz = da * self._relu_derivative(pre_activations[layer_idx])
            grads_w[layer_idx] = activations[layer_idx].T @ dz
            grads_b[layer_idx] = self.xp.sum(dz, axis=0, keepdims=True)

        for layer_idx in range(len(self.weights)):
            self.weights[layer_idx] -= learning_rate * grads_w[layer_idx]
            self.biases[layer_idx] -= learning_rate * grads_b[layer_idx]

        loss = self.cross_entropy_loss(probs, y_batch_one_hot)
        predictions = self.xp.argmax(probs, axis=1)
        labels = self.xp.argmax(y_batch_one_hot, axis=1)
        accuracy = self._to_float(self.xp.mean(predictions == labels))
        return loss, accuracy

    def predict_proba(self, x: Any) -> Any:
        probs, _ = self.forward(x)
        return probs

    def predict(self, x: Any) -> Any:
        return self.xp.argmax(self.predict_proba(x), axis=1)

    def evaluate(self, x: Any, y: Any) -> tuple[float, float]:
        probs = self.predict_proba(x)
        y_one_hot = one_hot(y, self.output_size, xp=self.xp)
        loss = self.cross_entropy_loss(probs, y_one_hot)
        acc = self._to_float(self.xp.mean(self.xp.argmax(probs, axis=1) == y))
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
                model.weights.append(model.asarray(data[w_key].astype(np.float32), dtype=model.xp.float32))
                model.biases.append(model.asarray(data[b_key].astype(np.float32), dtype=model.xp.float32))

            if not model.weights or not model.biases:
                model.weights = [
                    model.asarray(data["w1"].astype(np.float32), dtype=model.xp.float32),
                    model.asarray(data["w2"].astype(np.float32), dtype=model.xp.float32),
                ]
                model.biases = [
                    model.asarray(data["b1"].astype(np.float32), dtype=model.xp.float32),
                    model.asarray(data["b2"].astype(np.float32), dtype=model.xp.float32),
                ]
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

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        if dtype is None:
            return self.xp.asarray(value)
        return self.xp.asarray(value, dtype=dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self.backend == "gpu" and cp is not None:
            return cp.asnumpy(value)
        return np.asarray(value)

    def _to_float(self, value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
