from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deps import ensure_requirements_installed

ensure_requirements_installed()

import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
except Exception as exc:  # noqa: BLE001
    cl = None
    cl_array = None
    _PYOPENCL_IMPORT_ERROR = str(exc)
else:
    _PYOPENCL_IMPORT_ERROR = None


_SUM_AXIS0_KERNEL_SRC = """
__kernel void sum_axis0_f32(
    __global const float *x,
    __global float *out,
    const int rows,
    const int cols)
{
    int c = get_global_id(0);
    if (c >= cols) {
        return;
    }
    float acc = 0.0f;
    for (int r = 0; r < rows; ++r) {
        acc += x[r * cols + c];
    }
    out[c] = acc;
}
"""


def _probe_opencl_gpu() -> tuple[bool, str | None, dict[str, Any]]:
    if cl is None or cl_array is None:
        reason = _PYOPENCL_IMPORT_ERROR or "PyOpenCL nicht verfuegbar."
        return False, reason, {}

    try:
        platforms = cl.get_platforms()
        gpu_devices: list[Any] = []
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            for device in devices:
                gpu_devices.append((platform, device))

        if not gpu_devices:
            return False, "Keine OpenCL-GPU gefunden.", {}

        platform, device = gpu_devices[0]
        context = cl.Context(devices=[device])
        queue = cl.CommandQueue(context)
        program = cl.Program(context, _SUM_AXIS0_KERNEL_SRC).build()

        # kleiner Funktionstest fuer Dot-Operationen
        a = cl_array.to_device(queue, np.array([[1.0, 2.0]], dtype=np.float32))
        b = cl_array.to_device(queue, np.array([[3.0], [4.0]], dtype=np.float32))
        c = cl_array.dot(a, b)
        if abs(float(c.get()[0, 0]) - 11.0) > 1e-5:
            return False, "OpenCL-Dot-Test fehlgeschlagen.", {}

        backend_info = {
            "provider": "pyopencl",
            "runtime": "opencl",
            "platform": str(platform.name),
            "device": str(device.name),
            "vendor": str(device.vendor),
        }
        return True, None, {
            "context": context,
            "queue": queue,
            "program": program,
            "backend_info": backend_info,
        }
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), {}


def resolve_compute_backend(requested_backend: str) -> tuple[str, str | None, Any, dict[str, Any] | None, dict[str, Any]]:
    value = (requested_backend or "cpu").strip().lower()
    if value == "cpu":
        return "cpu", None, np, None, {"provider": "numpy", "runtime": "cpu"}

    if value != "gpu":
        return "cpu", f"Unbekannter Backend-Wert '{requested_backend}', nutze CPU.", np, None, {
            "provider": "numpy",
            "runtime": "cpu",
        }

    ok, reason, runtime = _probe_opencl_gpu()
    if not ok:
        return "cpu", f"GPU angefragt, aber OpenCL nicht nutzbar ({reason}). Nutze CPU.", np, None, {
            "provider": "numpy",
            "runtime": "cpu",
        }

    backend_info = dict(runtime.get("backend_info", {}))
    return "gpu", None, np, runtime, backend_info


def one_hot(labels: Any, num_classes: int = 10) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int64)
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
        self.backend, self.backend_note, self.xp, self._opencl_runtime, self.backend_info = resolve_compute_backend(backend)

        self.queue = self._opencl_runtime["queue"] if self._opencl_runtime is not None else None
        self._sum_axis0_kernel = (
            self._opencl_runtime["program"].sum_axis0_f32 if self._opencl_runtime is not None else None
        )

        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = max(1, int(hidden_layers))
        self.output_size = output_size

        layer_sizes = [self.input_size] + [self.hidden_size] * self.hidden_layers + [self.output_size]
        self.layer_sizes = layer_sizes

        self.weights: list[Any] = []
        self.biases: list[Any] = []
        for prev_size, next_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
            weight_np = rng.normal(0.0, np.sqrt(2.0 / prev_size), (prev_size, next_size)).astype(np.float32)
            bias_np = np.zeros((1, next_size), dtype=np.float32)
            self.weights.append(self.asarray(weight_np, dtype=np.float32))
            self.biases.append(self.asarray(bias_np, dtype=np.float32))

    def _is_opencl_array(self, value: Any) -> bool:
        return cl_array is not None and isinstance(value, cl_array.Array)

    def _relu(self, x: Any) -> Any:
        if self._is_opencl_array(x):
            return cl_array.maximum(x, np.float32(0.0))
        return np.maximum(0.0, x)

    def _relu_derivative(self, x: Any) -> Any:
        if self._is_opencl_array(x):
            return (x > 0.0).astype(np.float32)
        return (x > 0.0).astype(np.float32)

    def _softmax_numpy(self, x_np: np.ndarray) -> np.ndarray:
        shifted = x_np - np.max(x_np, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _softmax(self, x: Any) -> Any:
        if self._is_opencl_array(x):
            x_np = self.to_numpy(x)
            probs_np = self._softmax_numpy(x_np)
            return self.asarray(probs_np, dtype=np.float32)
        return self._softmax_numpy(np.asarray(x, dtype=np.float32))

    def _zeros_like(self, value: Any) -> Any:
        if self._is_opencl_array(value):
            return cl_array.zeros(self.queue, value.shape, value.dtype)
        return np.zeros_like(value)

    def _sum_axis0_keepdims(self, value: Any) -> Any:
        if self._is_opencl_array(value):
            rows = int(value.shape[0])
            cols = int(value.shape[1])
            out = cl_array.zeros(self.queue, (cols,), np.float32)
            self._sum_axis0_kernel(
                self.queue,
                (cols,),
                None,
                value.data,
                out.data,
                np.int32(rows),
                np.int32(cols),
            )
            return out.reshape((1, cols))
        return np.sum(value, axis=0, keepdims=True)

    def _matmul(self, a: Any, b: Any) -> Any:
        if self._is_opencl_array(a) and self._is_opencl_array(b):
            return cl_array.dot(a, b)
        return a @ b

    def forward(self, x: Any) -> tuple[Any, dict[str, Any]]:
        x = self.asarray(x, dtype=np.float32)
        activations: list[Any] = [x]
        pre_activations: list[Any] = []

        a = x
        for layer_idx in range(len(self.weights) - 1):
            z = self._matmul(a, self.weights[layer_idx]) + self.biases[layer_idx]
            a = self._relu(z)
            pre_activations.append(z)
            activations.append(a)

        z_out = self._matmul(a, self.weights[-1]) + self.biases[-1]
        probs = self._softmax(z_out)
        pre_activations.append(z_out)
        activations.append(probs)

        cache = {"activations": activations, "pre_activations": pre_activations, "probs": probs}
        return probs, cache

    def cross_entropy_loss(self, probs: Any, y_one_hot: Any) -> float:
        probs_np = self.to_numpy(probs)
        labels_np = self.to_numpy(y_one_hot)
        eps = 1e-12
        losses = -np.sum(labels_np * np.log(probs_np + eps), axis=1)
        return float(np.mean(losses))

    def train_batch(self, x_batch: Any, y_batch_one_hot: Any, learning_rate: float) -> tuple[float, float]:
        x_batch = self.asarray(x_batch, dtype=np.float32)
        y_batch_one_hot = self.asarray(y_batch_one_hot, dtype=np.float32)

        probs, cache = self.forward(x_batch)

        batch_size = int(x_batch.shape[0])
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]

        grads_w: list[Any] = [self._zeros_like(w) for w in self.weights]
        grads_b: list[Any] = [self._zeros_like(b) for b in self.biases]

        dz = (probs - y_batch_one_hot) / float(batch_size)
        last_idx = len(self.weights) - 1
        grads_w[last_idx] = self._matmul(activations[last_idx].T, dz)
        grads_b[last_idx] = self._sum_axis0_keepdims(dz)

        for layer_idx in range(last_idx - 1, -1, -1):
            da = self._matmul(dz, self.weights[layer_idx + 1].T)
            dz = da * self._relu_derivative(pre_activations[layer_idx])
            grads_w[layer_idx] = self._matmul(activations[layer_idx].T, dz)
            grads_b[layer_idx] = self._sum_axis0_keepdims(dz)

        for layer_idx in range(len(self.weights)):
            self.weights[layer_idx] = self.weights[layer_idx] - learning_rate * grads_w[layer_idx]
            self.biases[layer_idx] = self.biases[layer_idx] - learning_rate * grads_b[layer_idx]

        loss = self.cross_entropy_loss(probs, y_batch_one_hot)
        probs_np = self.to_numpy(probs)
        labels_np = self.to_numpy(y_batch_one_hot)
        predictions = np.argmax(probs_np, axis=1)
        labels = np.argmax(labels_np, axis=1)
        accuracy = float(np.mean(predictions == labels))
        return loss, accuracy

    def one_hot_labels(self, labels: Any, num_classes: int = 10) -> Any:
        encoded_np = one_hot(self.to_numpy(labels), num_classes=num_classes)
        return self.asarray(encoded_np, dtype=np.float32)

    def predict_proba(self, x: Any) -> Any:
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

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        if self.backend == "gpu" and self.queue is not None and cl_array is not None:
            if self._is_opencl_array(value):
                if dtype is None or value.dtype == dtype:
                    return value
                return value.astype(dtype)
            array_np = np.asarray(value, dtype=dtype if dtype is not None else np.float32)
            return cl_array.to_device(self.queue, array_np)

        return np.asarray(value, dtype=dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self._is_opencl_array(value):
            return value.get()
        return np.asarray(value)

    def _to_float(self, value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

