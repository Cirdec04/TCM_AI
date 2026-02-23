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
        output_size: int = 10,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_size), (input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_size), dtype=np.float32)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size)).astype(np.float32)
        self.b2 = np.zeros((1, output_size), dtype=np.float32)

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

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.w1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.w2 + self.b2
        probs = self._softmax(z2)
        cache = {"x": x, "z1": z1, "a1": a1, "probs": probs}
        return probs, cache

    @staticmethod
    def cross_entropy_loss(probs: np.ndarray, y_one_hot: np.ndarray) -> float:
        eps = 1e-12
        losses = -np.sum(y_one_hot * np.log(probs + eps), axis=1)
        return float(np.mean(losses))

    def train_batch(self, x_batch: np.ndarray, y_batch_one_hot: np.ndarray, learning_rate: float) -> tuple[float, float]:
        probs, cache = self.forward(x_batch)

        batch_size = x_batch.shape[0]
        dz2 = (probs - y_batch_one_hot) / batch_size
        dw2 = cache["a1"].T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * self._relu_derivative(cache["z1"])
        dw1 = cache["x"].T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

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
        np.savez(
            target,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            metadata=json.dumps(metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["SimpleMLP", dict[str, Any]]:
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            input_size = int(data["input_size"])
            hidden_size = int(data["hidden_size"])
            output_size = int(data["output_size"])
            model = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
            model.w1 = data["w1"].astype(np.float32)
            model.b1 = data["b1"].astype(np.float32)
            model.w2 = data["w2"].astype(np.float32)
            model.b2 = data["b2"].astype(np.float32)
            metadata_raw = str(data["metadata"])

        metadata: dict[str, Any]
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
        return model, metadata
