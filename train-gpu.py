from __future__ import annotations

import argparse
import json
import re
import threading
import tkinter as tk
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from tkinter import messagebox, ttk
from typing import Any, Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mpl_fig

from deps import ensure_requirements_installed

ensure_requirements_installed(required_modules=("numpy", "matplotlib", "PIL"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

cl = None
cl_array = None


def ensure_pyopencl() -> None:
    global cl, cl_array
    if cl is not None and cl_array is not None:
        return
    try:
        import pyopencl as _cl
        import pyopencl.array as _cl_array
    except ImportError as exc:
        raise SystemExit(
            "pyopencl fehlt. Bitte zuerst installieren, z. B. mit: pip install pyopencl"
        ) from exc
    cl = _cl
    cl_array = _cl_array


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "training"
TEST_DATA_DIR = DATA_DIR / "testing"
MODELS_DIR = BASE_DIR / "models"

MODEL_PROFILES = {
    "mini": {
        "hidden_size": 256,
        "hidden_layers": 2,
        "epochs": 96,
        "batch_size": 512,
        "learning_rate": 0.0025,
    },
    "normal": {
        "hidden_size": 512,
        "hidden_layers": 2,
        "epochs": 192,
        "batch_size": 512,
        "learning_rate": 0.0015,
    },
    "pro": {
        "hidden_size": 2048,
        "hidden_layers": 3,
        "epochs": 512,
        "batch_size": 512,
        "learning_rate": 0.001,
    },
}

ProgressCallback = Callable[[str, dict[str, Any]], None]


def _emit(callback: ProgressCallback | None, event: str, **data: Any) -> None:
    if callback is not None:
        callback(event, data)


OPENCL_KERNELS = r"""
#define TILE 16
#define NEG_INF (-3.402823466e+38F)

__kernel void matmul_bias_relu(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K,
    const int apply_relu
){
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float acc = 0.0f;
    const int tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * TILE + lx;
        const int b_row = t * TILE + ly;

        As[ly][lx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ly][lx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; ++k) {
            acc += As[ly][k] * Bs[k][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        acc += bias[col];
        if (apply_relu && acc < 0.0f) {
            acc = 0.0f;
        }
        C[row * N + col] = acc;
    }
}

__kernel void matmul_at_b(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K,
    const int N
){
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float acc = 0.0f;
    const int tiles = (M + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        const int m_from_x = t * TILE + lx;
        const int m_from_y = t * TILE + ly;

        As[lx][ly] = (m_from_x < M && row < K) ? A[m_from_x * K + row] : 0.0f;
        Bs[ly][lx] = (m_from_y < M && col < N) ? B[m_from_y * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int m = 0; m < TILE; ++m) {
            acc += As[m][ly] * Bs[m][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < K && col < N) {
        C[row * N + col] = acc;
    }
}

__kernel void matmul_a_bt(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
){
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float acc = 0.0f;
    const int tiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        const int n_for_a = t * TILE + lx;
        const int n_for_b = t * TILE + ly;

        As[ly][lx] = (row < M && n_for_a < N) ? A[row * N + n_for_a] : 0.0f;
        Bs[ly][lx] = (col < K && n_for_b < N) ? B[col * N + n_for_b] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int n = 0; n < TILE; ++n) {
            acc += As[ly][n] * Bs[n][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

__kernel void softmax_xent_grad(
    __global const float* logits,
    __global const int* labels,
    __global float* dz,
    __global float* sample_loss,
    __global int* sample_correct,
    const int M,
    const int C,
    const float inv_batch
){
    const int i = get_global_id(0);
    if (i >= M) {
        return;
    }

    const int base = i * C;
    float max_v = NEG_INF;
    for (int c = 0; c < C; ++c) {
        const float value = logits[base + c];
        max_v = value > max_v ? value : max_v;
    }

    float denom = 0.0f;
    for (int c = 0; c < C; ++c) {
        denom += exp(logits[base + c] - max_v);
    }

    const int label = labels[i];
    float label_prob = 1e-12f;
    float best_prob = NEG_INF;
    int best_class = 0;

    for (int c = 0; c < C; ++c) {
        const float p = exp(logits[base + c] - max_v) / denom;
        if (c == label) {
            label_prob = p > 1e-12f ? p : 1e-12f;
        }
        dz[base + c] = (p - (c == label ? 1.0f : 0.0f)) * inv_batch;
        if (p > best_prob) {
            best_prob = p;
            best_class = c;
        }
    }

    sample_loss[i] = -log(label_prob);
    sample_correct[i] = (best_class == label) ? 1 : 0;
}

__kernel void softmax_metrics(
    __global const float* logits,
    __global const int* labels,
    __global float* sample_loss,
    __global int* sample_correct,
    const int M,
    const int C
){
    const int i = get_global_id(0);
    if (i >= M) {
        return;
    }

    const int base = i * C;
    float max_v = NEG_INF;
    for (int c = 0; c < C; ++c) {
        const float value = logits[base + c];
        max_v = value > max_v ? value : max_v;
    }

    float denom = 0.0f;
    for (int c = 0; c < C; ++c) {
        denom += exp(logits[base + c] - max_v);
    }

    const int label = labels[i];
    float label_prob = 1e-12f;
    float best_prob = NEG_INF;
    int best_class = 0;

    for (int c = 0; c < C; ++c) {
        const float p = exp(logits[base + c] - max_v) / denom;
        if (c == label) {
            label_prob = p > 1e-12f ? p : 1e-12f;
        }
        if (p > best_prob) {
            best_prob = p;
            best_class = c;
        }
    }

    sample_loss[i] = -log(label_prob);
    sample_correct[i] = (best_class == label) ? 1 : 0;
}

__kernel void reduce_sum_rows(
    __global const float* A,
    __global float* out,
    const int M,
    const int N
){
    const int col = get_global_id(0);
    if (col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int row = 0; row < M; ++row) {
        acc += A[row * N + col];
    }
    out[col] = acc;
}

__kernel void relu_backprop_inplace(
    __global float* da,
    __global const float* activation,
    const int size
){
    const int idx = get_global_id(0);
    if (idx >= size) {
        return;
    }
    if (activation[idx] <= 0.0f) {
        da[idx] = 0.0f;
    }
}

__kernel void sgd_update(
    __global float* param,
    __global const float* grad,
    const float learning_rate,
    const int size
){
    const int idx = get_global_id(0);
    if (idx >= size) {
        return;
    }
    param[idx] -= learning_rate * grad[idx];
}

__kernel void gather_features(
    __global const float* full_data,
    __global const int* indices,
    __global float* out_batch,
    const int num_indices,
    const int feature_dim
){
    const int i = get_global_id(0);
    const int f = get_global_id(1);
    if (i < num_indices && f < feature_dim) {
        const int src_idx = indices[i];
        out_batch[i * feature_dim + f] = full_data[src_idx * feature_dim + f];
    }
}

__kernel void gather_labels(
    __global const int* full_labels,
    __global const int* indices,
    __global int* out_batch,
    const int num_indices
){
    const int i = get_global_id(0);
    if (i < num_indices) {
        const int src_idx = indices[i];
        out_batch[i] = full_labels[src_idx];
    }
}
"""


@dataclass(frozen=True)
class OpenCLDeviceRef:
    platform_index: int
    device_index: int
    platform_name: str
    device_name: str
    vendor: str
    is_gpu: bool
    compute_units: int
    global_mem_mb: int
    driver_version: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-Training (OpenCL) fuer TCM-AI.")
    parser.add_argument("--size", choices=["mini", "normal", "pro"], default="normal")
    parser.add_argument("--version", type=str, default=None, help="Versionsnummer des Modells (z. B. 4.1).")
    parser.add_argument("--list-devices", action="store_true", help="Zeigt OpenCL Plattformen/Geraete und beendet.")
    parser.add_argument("--platform-index", type=int, default=None, help="OpenCL Plattform-Index.")
    parser.add_argument("--device-index", type=int, default=None, help="OpenCL Geraete-Index innerhalb der Plattform.")
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="Ueberschreibt die Batch-Size aus dem Profil (groesser = oft schneller auf GPU).",
    )
    parser.add_argument("--test-eval-interval", type=int, default=1, help="Test-Auswertung alle N Epochen.")
    parser.add_argument("--no-fast-math", action="store_true", help="Deaktiviert OpenCL fast math Build-Optionen.")
    parser.add_argument("--no-ui", action="store_true", help="Kein GUI, direkt im Terminal trainieren.")
    return parser.parse_args()


def _round_up(value: int, multiple: int) -> int:
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def _load_image_as_vector(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (28, 28):
            image = image.resize((28, 28))
        pixels = np.asarray(image, dtype=np.float32)
    pixels = pixels / 255.0
    return pixels.reshape(-1)


def load_dataset_from_folders(
    data_dir: Path,
    dataset_label: str,
    callback: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    files_with_labels: list[tuple[int, Path]] = []

    for class_label in range(10):
        class_dir = data_dir / str(class_label)
        if not class_dir.exists():
            continue
        files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        for file_path in files:
            files_with_labels.append((class_label, file_path))

    total_files = len(files_with_labels)
    _emit(callback, "info", message=f"{dataset_label}: {total_files} Dateien gefunden.")
    for idx, (digit_label, file_path) in enumerate(files_with_labels, start=1):
        features.append(_load_image_as_vector(file_path))
        labels.append(digit_label)
        if idx == 1 or idx % progress_every == 0 or idx == total_files:
            _emit(callback, "info", message=f"{dataset_label}: {idx}/{total_files} geladen.")

    x = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    return x, y


def validate_version(version: str) -> str:
    value = version.strip()
    if not re.fullmatch(r"\d+(?:\.\d+)*", value):
        raise ValueError("Version muss wie 1, 2.1 oder 2.1.3 aussehen.")
    return value


def build_model_name(version: str, size: str) -> str:
    suffix = "" if size == "normal" else f"-{size}"
    return f"TCM-o{version}{suffix}"


def get_fixed_seed() -> int:
    return 42


def enumerate_opencl_devices() -> list[OpenCLDeviceRef]:
    ensure_pyopencl()
    refs: list[OpenCLDeviceRef] = []
    platforms = cl.get_platforms()
    for p_idx, platform in enumerate(platforms):
        devices = platform.get_devices()
        for d_idx, device in enumerate(devices):
            is_gpu = bool(device.type & cl.device_type.GPU)
            refs.append(
                OpenCLDeviceRef(
                    platform_index=p_idx,
                    device_index=d_idx,
                    platform_name=str(platform.name).strip(),
                    device_name=str(device.name).strip(),
                    vendor=str(device.vendor).strip(),
                    is_gpu=is_gpu,
                    compute_units=int(device.max_compute_units),
                    global_mem_mb=int(device.global_mem_size // (1024 * 1024)),
                    driver_version=str(device.driver_version).strip(),
                )
            )
    return refs


def _pick_device(refs: list[OpenCLDeviceRef], platform_index: int | None, device_index: int | None) -> OpenCLDeviceRef:
    if not refs:
        raise RuntimeError("Keine OpenCL Geraete gefunden.")

    if platform_index is not None or device_index is not None:
        if platform_index is None or device_index is None:
            raise ValueError("--platform-index und --device-index muessen zusammen gesetzt werden.")
        for ref in refs:
            if ref.platform_index == platform_index and ref.device_index == device_index:
                return ref
        raise ValueError(f"OpenCL Geraet [{platform_index}:{device_index}] wurde nicht gefunden.")

    gpu_refs = [ref for ref in refs if ref.is_gpu]
    candidates = gpu_refs if gpu_refs else refs

    def score(ref: OpenCLDeviceRef) -> int:
        vendor = ref.vendor.lower()
        value = 0
        if ref.is_gpu:
            value += 1000
        if "amd" in vendor or "advanced micro devices" in vendor:
            value += 200
        elif "nvidia" in vendor:
            value += 150
        elif "intel" in vendor:
            value += 100
        value += ref.compute_units
        value += ref.global_mem_mb // 256
        return value

    return max(candidates, key=score)


def create_opencl_context(
    platform_index: int | None,
    device_index: int | None,
    fast_math: bool,
) -> tuple[cl.Context, cl.CommandQueue, cl.Program, OpenCLDeviceRef]:
    ensure_pyopencl()
    refs = enumerate_opencl_devices()
    selected = _pick_device(refs, platform_index=platform_index, device_index=device_index)
    platform = cl.get_platforms()[selected.platform_index]
    device = platform.get_devices()[selected.device_index]

    context = cl.Context(devices=[device])
    queue = cl.CommandQueue(context, properties=0)

    build_options: list[str] = []
    if fast_math:
        build_options.extend(["-cl-fast-relaxed-math", "-cl-mad-enable"])

    try:
        program = cl.Program(context, OPENCL_KERNELS).build(options=build_options)
    except Exception as exc:  # noqa: BLE001
        build_log = ""
        if hasattr(exc, "build_log"):
            log_parts = []
            for _dev, log in getattr(exc, "build_log"):
                log_parts.append(str(log))
            build_log = "\n".join(log_parts)
        if build_log:
            raise RuntimeError(f"OpenCL Kernel Build fehlgeschlagen:\n{build_log}") from exc
        raise

    return context, queue, program, selected


class OpenCLMLPTrainer:
    def __init__(
        self,
        queue: cl.CommandQueue,
        program: cl.Program,
        layer_sizes: list[int],
        batch_size: int,
        seed: int,
    ) -> None:
        self.queue = queue
        self.program = program
        self.layer_sizes = [int(v) for v in layer_sizes]
        self.batch_size = int(batch_size)
        self.num_layers = len(self.layer_sizes) - 1
        self.hidden_layers = max(0, self.num_layers - 1)
        self.hidden_size = self.layer_sizes[1] if self.hidden_layers > 0 else 0
        self.output_size = self.layer_sizes[-1]
        self.local_2d = (16, 16)
        self.local_1d = (256,)

        rng = np.random.default_rng(seed)
        self.weights: list[cl_array.Array] = []
        self.biases: list[cl_array.Array] = []
        self.grad_weights: list[cl_array.Array] = []
        self.grad_biases: list[cl_array.Array] = []

        for prev_size, next_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=False):
            weight_np = rng.normal(0.0, np.sqrt(2.0 / prev_size), (prev_size, next_size)).astype(np.float32)
            bias_np = np.zeros((next_size,), dtype=np.float32)
            self.weights.append(cl_array.to_device(self.queue, weight_np))
            self.biases.append(cl_array.to_device(self.queue, bias_np))
            self.grad_weights.append(cl_array.empty(self.queue, weight_np.shape, dtype=np.float32))
            self.grad_biases.append(cl_array.empty(self.queue, bias_np.shape, dtype=np.float32))

        self.activations: list[cl_array.Array] = []
        for size in self.layer_sizes[:-1]:
            self.activations.append(cl_array.empty(self.queue, (self.batch_size, size), dtype=np.float32))
        self.logits = cl_array.empty(self.queue, (self.batch_size, self.output_size), dtype=np.float32)
        self.dz_output = cl_array.empty(self.queue, (self.batch_size, self.output_size), dtype=np.float32)

        if self.hidden_layers > 0:
            self.backprop_a = cl_array.empty(self.queue, (self.batch_size, self.hidden_size), dtype=np.float32)
            self.backprop_b = cl_array.empty(self.queue, (self.batch_size, self.hidden_size), dtype=np.float32)
        else:
            self.backprop_a = None
            self.backprop_b = None

        self.labels = cl_array.empty(self.queue, (self.batch_size,), dtype=np.int32)
        self.sample_loss = cl_array.empty(self.queue, (self.batch_size,), dtype=np.float32)
        self.sample_correct = cl_array.empty(self.queue, (self.batch_size,), dtype=np.int32)

        self.host_loss = np.empty((self.batch_size,), dtype=np.float32)
        self.host_correct = np.empty((self.batch_size,), dtype=np.int32)

    def _k_matmul_bias_relu(
        self,
        a: cl_array.Array,
        b: cl_array.Array,
        bias: cl_array.Array,
        out: cl_array.Array,
        m: int,
        n: int,
        k: int,
        apply_relu: int,
    ) -> None:
        global_size = (_round_up(n, self.local_2d[0]), _round_up(m, self.local_2d[1]))
        self.program.matmul_bias_relu(
            self.queue,
            global_size,
            self.local_2d,
            a.data,
            b.data,
            bias.data,
            out.data,
            np.int32(m),
            np.int32(n),
            np.int32(k),
            np.int32(apply_relu),
        )

    def _k_matmul_at_b(
        self,
        a: cl_array.Array,
        b: cl_array.Array,
        out: cl_array.Array,
        m: int,
        k: int,
        n: int,
    ) -> None:
        global_size = (_round_up(n, self.local_2d[0]), _round_up(k, self.local_2d[1]))
        self.program.matmul_at_b(
            self.queue,
            global_size,
            self.local_2d,
            a.data,
            b.data,
            out.data,
            np.int32(m),
            np.int32(k),
            np.int32(n),
        )

    def _k_matmul_a_bt(
        self,
        a: cl_array.Array,
        b: cl_array.Array,
        out: cl_array.Array,
        m: int,
        n: int,
        k: int,
    ) -> None:
        global_size = (_round_up(k, self.local_2d[0]), _round_up(m, self.local_2d[1]))
        self.program.matmul_a_bt(
            self.queue,
            global_size,
            self.local_2d,
            a.data,
            b.data,
            out.data,
            np.int32(m),
            np.int32(n),
            np.int32(k),
        )

    def _k_softmax_xent(self, m: int) -> None:
        inv_batch = np.float32(1.0 / float(m))
        global_size = (_round_up(m, self.local_1d[0]),)
        self.program.softmax_xent_grad(
            self.queue,
            global_size,
            self.local_1d,
            self.logits.data,
            self.labels.data,
            self.dz_output.data,
            self.sample_loss.data,
            self.sample_correct.data,
            np.int32(m),
            np.int32(self.output_size),
            inv_batch,
        )

    def _k_softmax_metrics(self, m: int) -> None:
        global_size = (_round_up(m, self.local_1d[0]),)
        self.program.softmax_metrics(
            self.queue,
            global_size,
            self.local_1d,
            self.logits.data,
            self.labels.data,
            self.sample_loss.data,
            self.sample_correct.data,
            np.int32(m),
            np.int32(self.output_size),
        )

    def _k_reduce_sum_rows(self, a: cl_array.Array, out: cl_array.Array, m: int, n: int) -> None:
        global_size = (_round_up(n, self.local_1d[0]),)
        self.program.reduce_sum_rows(
            self.queue,
            global_size,
            self.local_1d,
            a.data,
            out.data,
            np.int32(m),
            np.int32(n),
        )

    def _k_relu_backprop_inplace(self, da: cl_array.Array, activation: cl_array.Array, size: int) -> None:
        global_size = (_round_up(size, self.local_1d[0]),)
        self.program.relu_backprop_inplace(
            self.queue,
            global_size,
            self.local_1d,
            da.data,
            activation.data,
            np.int32(size),
        )

    def _k_sgd_update(self, param: cl_array.Array, grad: cl_array.Array, learning_rate: np.float32, size: int) -> None:
        global_size = (_round_up(size, self.local_1d[0]),)
        self.program.sgd_update(
            self.queue,
            global_size,
            self.local_1d,
            param.data,
            grad.data,
            learning_rate,
            np.int32(size),
        )

    def _k_gather_batch(
        self,
        full_x: cl_array.Array,
        full_y: cl_array.Array,
        indices_gpu: cl_array.Array,
        m: int,
    ) -> None:
        # Features gather
        feat_dim = self.layer_sizes[0]
        global_feat = (_round_up(m, 16), _round_up(feat_dim, 16))
        local_feat = (16, 16)
        self.program.gather_features(
            self.queue,
            global_feat,
            local_feat,
            full_x.data,
            indices_gpu.data,
            self.activations[0].data,
            np.int32(m),
            np.int32(feat_dim),
        )
        # Labels gather
        global_lab = (_round_up(m, 256),)
        local_lab = (256,)
        self.program.gather_labels(
            self.queue,
            global_lab,
            local_lab,
            full_y.data,
            indices_gpu.data,
            self.labels.data,
            np.int32(m),
        )

    def _forward(self, m: int) -> None:
        for layer_idx in range(self.hidden_layers):
            prev_size = self.layer_sizes[layer_idx]
            next_size = self.layer_sizes[layer_idx + 1]
            self._k_matmul_bias_relu(
                self.activations[layer_idx],
                self.weights[layer_idx],
                self.biases[layer_idx],
                self.activations[layer_idx + 1],
                m=m,
                n=next_size,
                k=prev_size,
                apply_relu=1,
            )

        output_layer_idx = self.num_layers - 1
        pre_output_activation = self.activations[self.hidden_layers]
        pre_output_size = self.layer_sizes[-2]
        self._k_matmul_bias_relu(
            pre_output_activation,
            self.weights[output_layer_idx],
            self.biases[output_layer_idx],
            self.logits,
            m=m,
            n=self.output_size,
            k=pre_output_size,
            apply_relu=0,
        )

    def _backward_and_update(self, m: int, learning_rate: float) -> None:
        lr = np.float32(learning_rate)
        last = self.num_layers - 1

        pre_output_activation = self.activations[self.hidden_layers]
        pre_output_size = self.layer_sizes[-2]
        self._k_matmul_at_b(
            pre_output_activation,
            self.dz_output,
            self.grad_weights[last],
            m=m,
            k=pre_output_size,
            n=self.output_size,
        )
        self._k_reduce_sum_rows(self.dz_output, self.grad_biases[last], m=m, n=self.output_size)

        if self.hidden_layers > 0 and self.backprop_a is not None and self.backprop_b is not None:
            self._k_matmul_a_bt(
                self.dz_output,
                self.weights[last],
                self.backprop_a,
                m=m,
                n=self.output_size,
                k=self.hidden_size,
            )
            da_current = self.backprop_a
            da_next = self.backprop_b

            for hidden_idx in range(self.hidden_layers - 1, -1, -1):
                hidden_width = self.layer_sizes[hidden_idx + 1]
                self._k_relu_backprop_inplace(
                    da_current,
                    self.activations[hidden_idx + 1],
                    size=m * hidden_width,
                )
                prev_width = self.layer_sizes[hidden_idx]
                self._k_matmul_at_b(
                    self.activations[hidden_idx],
                    da_current,
                    self.grad_weights[hidden_idx],
                    m=m,
                    k=prev_width,
                    n=hidden_width,
                )
                self._k_reduce_sum_rows(
                    da_current,
                    self.grad_biases[hidden_idx],
                    m=m,
                    n=hidden_width,
                )

                if hidden_idx > 0:
                    prev_hidden_width = self.layer_sizes[hidden_idx]
                    self._k_matmul_a_bt(
                        da_current,
                        self.weights[hidden_idx],
                        da_next,
                        m=m,
                        n=hidden_width,
                        k=prev_hidden_width,
                    )
                    da_current, da_next = da_next, da_current

        for layer_idx in range(self.num_layers):
            self._k_sgd_update(
                self.weights[layer_idx],
                self.grad_weights[layer_idx],
                learning_rate=lr,
                size=self.weights[layer_idx].size,
            )
            self._k_sgd_update(
                self.biases[layer_idx],
                self.grad_biases[layer_idx],
                learning_rate=lr,
                size=self.biases[layer_idx].size,
            )

    def train_batch_vram(
        self,
        full_x_gpu: cl_array.Array,
        full_y_gpu: cl_array.Array,
        indices_gpu: cl_array.Array,
        m: int,
        learning_rate: float,
    ) -> tuple[float, float]:
        if m <= 0:
            return 0.0, 0.0

        self._k_gather_batch(full_x_gpu, full_y_gpu, indices_gpu, m)
        self._forward(m)
        self._k_softmax_xent(m)

        self.sample_loss.get(queue=self.queue, ary=self.host_loss)
        self.sample_correct.get(queue=self.queue, ary=self.host_correct)
        batch_loss = float(np.mean(self.host_loss[:m]))
        batch_acc = float(np.mean(self.host_correct[:m]))

        self._backward_and_update(m, learning_rate=learning_rate)
        return batch_loss, batch_acc

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray, learning_rate: float) -> tuple[float, float]:
        m = int(x_batch.shape[0])
        if m <= 0:
            return 0.0, 0.0

        self.activations[0][:m].set(np.asarray(x_batch, dtype=np.float32), queue=self.queue)
        self.labels[:m].set(np.asarray(y_batch, dtype=np.int32), queue=self.queue)

        self._forward(m)
        self._k_softmax_xent(m)

        self.sample_loss.get(queue=self.queue, ary=self.host_loss)
        self.sample_correct.get(queue=self.queue, ary=self.host_correct)
        batch_loss = float(np.mean(self.host_loss[:m]))
        batch_acc = float(np.mean(self.host_correct[:m]))

        self._backward_and_update(m, learning_rate=learning_rate)
        return batch_loss, batch_acc

    def evaluate(self, x_eval: np.ndarray, y_eval: np.ndarray) -> tuple[float, float]:
        total_loss = 0.0
        total_correct = 0
        total_samples = int(x_eval.shape[0])

        for start in range(0, total_samples, self.batch_size):
            end = min(start + self.batch_size, total_samples)
            m = end - start
            if m <= 0:
                continue

            self.activations[0][:m].set(np.asarray(x_eval[start:end], dtype=np.float32), queue=self.queue)
            self.labels[:m].set(np.asarray(y_eval[start:end], dtype=np.int32), queue=self.queue)

            self._forward(m)
            self._k_softmax_metrics(m)

            self.sample_loss.get(queue=self.queue, ary=self.host_loss)
            self.sample_correct.get(queue=self.queue, ary=self.host_correct)
            total_loss += float(np.sum(self.host_loss[:m], dtype=np.float64))
            total_correct += int(np.sum(self.host_correct[:m], dtype=np.int64))

        if total_samples <= 0:
            return 0.0, 0.0
        return total_loss / float(total_samples), total_correct / float(total_samples)

    def evaluate_vram(self, x_eval_gpu: cl_array.Array, y_eval_gpu: cl_array.Array) -> tuple[float, float]:
        total_loss = 0.0
        total_correct = 0
        total_samples = int(x_eval_gpu.shape[0])

        # Wir brauchen temporäre Indizes für die Auswertung
        eval_indices = np.arange(total_samples, dtype=np.int32)
        indices_gpu = cl_array.empty(self.queue, (self.batch_size,), dtype=np.int32)

        for start in range(0, total_samples, self.batch_size):
            end = min(start + self.batch_size, total_samples)
            m = end - start
            if m <= 0:
                continue

            indices_gpu[:m].set(eval_indices[start:end], queue=self.queue)
            self._k_gather_batch(x_eval_gpu, y_eval_gpu, indices_gpu, m)
            self._forward(m)
            self._k_softmax_metrics(m)

            self.sample_loss.get(queue=self.queue, ary=self.host_loss)
            self.sample_correct.get(queue=self.queue, ary=self.host_correct)
            total_loss += float(np.sum(self.host_loss[:m], dtype=np.float64))
            total_correct += int(np.sum(self.host_correct[:m], dtype=np.int64))

        if total_samples <= 0:
            return 0.0, 0.0
        return total_loss / float(total_samples), total_correct / float(total_samples)

    def export_weights(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for weight, bias in zip(self.weights, self.biases, strict=False):
            weights.append(weight.get(queue=self.queue).astype(np.float32, copy=False))
            biases.append(bias.get(queue=self.queue).astype(np.float32, copy=False))
        return weights, biases


def save_training_plot(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    epoch_count = len(epochs)
    x_ticks = np.linspace(1, max(1, epoch_count), num=6, dtype=int)
    x_ticks = np.unique(x_ticks)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Verlauf")
    plt.xlim(1, max(1, epoch_count))
    plt.xticks(x_ticks)
    plt.ylim(0.0, 2.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Verlauf")
    plt.xlim(1, max(1, epoch_count))
    plt.xticks(x_ticks)
    plt.ylim(0.0, 1.0)
    plt.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()


def save_model_json(metadata: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_model_npz(
    model_path: Path,
    layer_sizes: list[int],
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    metadata: dict[str, object],
) -> None:
    payload: dict[str, Any] = {
        "input_size": np.array(layer_sizes[0], dtype=np.int32),
        "hidden_size": np.array(layer_sizes[1], dtype=np.int32),
        "hidden_layers": np.array(len(layer_sizes) - 2, dtype=np.int32),
        "output_size": np.array(layer_sizes[-1], dtype=np.int32),
        "layer_sizes": np.array(layer_sizes, dtype=np.int32),
        "metadata": json.dumps(metadata),
    }
    for idx, (weight, bias) in enumerate(zip(weights, biases, strict=False), start=1):
        payload[f"w{idx}"] = np.asarray(weight, dtype=np.float32)
        payload[f"b{idx}"] = np.asarray(bias, dtype=np.float32).reshape(1, -1)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_path, **payload)


def print_device_list() -> None:
    refs = enumerate_opencl_devices()
    if not refs:
        print("Keine OpenCL Geraete gefunden.")
        return
    print("Gefundene OpenCL Geraete:")
    for ref in refs:
        dtype = "GPU" if ref.is_gpu else "CPU/Other"
        print(
            f"  [{ref.platform_index}:{ref.device_index}] {dtype} | {ref.vendor} | "
            f"{ref.platform_name} | {ref.device_name} | CU={ref.compute_units} | VRAM={ref.global_mem_mb} MB"
        )


def train_model_gpu(
    size: str,
    version: str,
    platform_index: int | None,
    device_index: int | None,
    batch_size_override: int | None,
    test_eval_interval: int,
    fast_math: bool,
    callback: ProgressCallback | None = None,
) -> dict[str, object]:
    if size not in MODEL_PROFILES:
        raise ValueError("Ungueltige Groesse. Erlaubt: mini, normal, pro.")
    version = validate_version(version)
    if test_eval_interval <= 0:
        raise ValueError("--test-eval-interval muss >= 1 sein.")

    profile = MODEL_PROFILES[size]
    hidden_size = int(profile["hidden_size"])
    hidden_layers = int(profile["hidden_layers"])
    epochs = int(profile["epochs"])
    base_batch_size = int(profile["batch_size"])
    learning_rate = float(profile["learning_rate"])
    seed = get_fixed_seed()

    effective_batch_size = int(batch_size_override) if batch_size_override is not None else base_batch_size
    if effective_batch_size <= 0:
        raise ValueError("Batch-Size muss groesser als 0 sein.")

    model_name = build_model_name(version=version, size=size)
    model_path = MODELS_DIR / f"{model_name}.npz"
    plot_path = MODELS_DIR / f"{model_name}_training.png"
    metadata_path = MODELS_DIR / f"{model_name}.json"

    if model_path.exists() or metadata_path.exists() or plot_path.exists():
        raise FileExistsError(
            f"Artefakt fuer '{model_name}' existiert bereits (.npz/.json/.png). Bitte andere Version waehlen."
        )

    _emit(callback, "info", message="Initialisiere OpenCL und bereite Kernel vor...")
    context, queue, program, selected_device = create_opencl_context(
        platform_index=platform_index,
        device_index=device_index,
        fast_math=fast_math,
    )
    _ = context

    _emit(callback, "info", message=f"Lade Trainingsdaten aus: {TRAIN_DATA_DIR}")
    x_train, y_train = load_dataset_from_folders(TRAIN_DATA_DIR, dataset_label="Training", callback=callback)
    _emit(callback, "info", message=f"Geladene Trainings-Samples: {len(y_train)}")

    _emit(callback, "info", message=f"Lade Testdaten aus: {TEST_DATA_DIR}")
    x_test, y_test = load_dataset_from_folders(TEST_DATA_DIR, dataset_label="Testing", callback=callback)
    _emit(callback, "info", message=f"Geladene Test-Samples: {len(y_test)}")

    layer_sizes = [28 * 28] + [hidden_size] * hidden_layers + [10]
    trainer = OpenCLMLPTrainer(
        queue=queue,
        program=program,
        layer_sizes=layer_sizes,
        batch_size=effective_batch_size,
        seed=seed,
    )

    _emit(
        callback,
        "info",
        message=(
            "OpenCL Geraet: "
            f"[{selected_device.platform_index}:{selected_device.device_index}] "
            f"{selected_device.vendor} | {selected_device.device_name} | CU={selected_device.compute_units}"
        ),
    )
    _emit(
        callback,
        "info",
        message=(
            "Training startet mit: "
            f"size={size}, hidden_size={hidden_size}, hidden_layers={hidden_layers}, epochs={epochs}, "
            f"batch_size={base_batch_size}, effective_batch_size={effective_batch_size}, "
            f"learning_rate={learning_rate}, seed={seed}, fast_math={fast_math}"
        ),
    )

    _emit(callback, "start", epochs=epochs)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    train_count = int(x_train.shape[0])
    num_batches = (train_count + effective_batch_size - 1) // effective_batch_size
    progress_every = max(1, num_batches // 4)

    _emit(callback, "info", message="Übertrage vollständigen Datensatz in den VRAM...")
    x_train_gpu = cl_array.to_device(queue, x_train)
    y_train_gpu = cl_array.to_device(queue, y_train.astype(np.int32))
    x_test_gpu = cl_array.to_device(queue, x_test)
    y_test_gpu = cl_array.to_device(queue, y_test.astype(np.int32))
    indices_gpu = cl_array.empty(queue, (effective_batch_size,), dtype=np.int32)

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(train_count).astype(np.int32)
        batch_losses: list[float] = []
        batch_accs: list[float] = []

        for batch_idx, start in enumerate(range(0, train_count, effective_batch_size), start=1):
            end = min(start + effective_batch_size, train_count)
            m = end - start
            batch_indices = indices[start:end]
            
            # Nur die Indizes zur GPU schicken (kleiner Overhead)
            indices_gpu[:m].set(batch_indices, queue=queue)

            batch_loss, batch_acc = trainer.train_batch_vram(
                x_train_gpu, y_train_gpu, indices_gpu, m, learning_rate=learning_rate
            )
            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)

            if batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == num_batches:
                _emit(
                    callback,
                    "info",
                    message=f"Epoch {epoch}/{epochs}: Batch {batch_idx}/{num_batches}",
                )

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        train_acc = float(np.mean(batch_accs)) if batch_accs else 0.0

        should_eval_test = (epoch == 1) or (epoch % test_eval_interval == 0) or (epoch == epochs)
        if should_eval_test:
            test_loss, test_acc = trainer.evaluate_vram(x_test_gpu, y_test_gpu)
        else:
            test_loss = float(history["test_loss"][-1]) if history["test_loss"] else 0.0
            test_acc = float(history["test_acc"][-1]) if history["test_acc"] else 0.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        _emit(
            callback,
            "progress",
            epoch=epoch,
            epochs=epochs,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )

    training_time_seconds = float(time.perf_counter() - start_time)
    weights, biases = trainer.export_weights()

    metadata: dict[str, object] = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_data_dir": str(TRAIN_DATA_DIR),
        "test_data_dir": str(TEST_DATA_DIR),
        "size": size,
        "hidden_size": hidden_size,
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "batch_size": base_batch_size,
        "effective_batch_size": effective_batch_size,
        "test_eval_interval": test_eval_interval,
        "learning_rate": learning_rate,
        "seed": seed,
        "compute_backend": "opencl",
        "backend_info": {
            "platform_index": selected_device.platform_index,
            "device_index": selected_device.device_index,
            "platform_name": selected_device.platform_name,
            "device_name": selected_device.device_name,
            "vendor": selected_device.vendor,
            "driver_version": selected_device.driver_version,
            "compute_units": selected_device.compute_units,
            "global_mem_mb": selected_device.global_mem_mb,
            "fast_math": bool(fast_math),
        },
        "samples": {"total": int(len(y_train) + len(y_test)), "train": int(len(y_train)), "test": int(len(y_test))},
        "final_metrics": {
            "train_loss": float(history["train_loss"][-1]),
            "train_acc": float(history["train_acc"][-1]),
            "test_loss": float(history["test_loss"][-1]),
            "test_acc": float(history["test_acc"][-1]),
        },
        "training_time_seconds": training_time_seconds,
        "artifacts": {
            "model_file": str(model_path),
            "plot_file": str(plot_path),
            "metadata_file": str(metadata_path),
        },
    }

    save_model_npz(
        model_path=model_path,
        layer_sizes=layer_sizes,
        weights=weights,
        biases=biases,
        metadata=metadata,
    )
    save_training_plot(history, plot_path)
    save_model_json(metadata, metadata_path)

    _emit(callback, "info", message="Training fertig.")
    _emit(callback, "info", message=f"Modell gespeichert: {model_path}")
    _emit(callback, "info", message=f"Plot gespeichert:   {plot_path}")
    _emit(callback, "info", message=f"JSON gespeichert:   {metadata_path}")
    return metadata


def run_cli(args: argparse.Namespace) -> None:
    def callback(event: str, data: dict[str, Any]) -> None:
        if event == "progress":
            print(
                f"Epoch {data['epoch']:02d}/{data['epochs']} | "
                f"Train Loss: {data['train_loss']:.4f} | Train Acc: {data['train_acc']:.4f} | "
                f"Test Loss: {data['test_loss']:.4f} | Test Acc: {data['test_acc']:.4f}"
            )
        elif event == "info":
            print(data["message"])

    metadata = train_model_gpu(
        size=args.size,
        version=args.version,
        platform_index=args.platform_index,
        device_index=args.device_index,
        batch_size_override=args.batch_size_override,
        test_eval_interval=args.test_eval_interval,
        fast_math=not args.no_fast_math,
        callback=callback,
    )
    final_acc = float((metadata.get("final_metrics", {}) or {}).get("test_acc", 0.0))
    print(f"Finale Test-Accuracy: {final_acc:.4f}")


class TrainingUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GPU Training Konfiguration (OpenCL)")
        self.root.resizable(False, False)

        self.event_queue: Queue[tuple[str, dict[str, Any]]] = Queue()
        self.training_running = False

        self.size_var = tk.StringVar(value="normal")
        self.version_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Bereit")

        # Live-Plot Daten
        self.history_data = {"epochs": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        self._build_ui()
        self._refresh_profile_label()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Modellgroesse:").grid(row=0, column=0, sticky="w")
        self.size_combo = ttk.Combobox(frame, textvariable=self.size_var, state="readonly", values=["mini", "normal", "pro"], width=12)
        self.size_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.size_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_profile_label())

        ttk.Label(frame, text="Version:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.version_entry = ttk.Entry(frame, textvariable=self.version_var, width=14)
        self.version_entry.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        self.profile_label = ttk.Label(frame, text="", justify="left")
        self.profile_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.start_btn = ttk.Button(frame, text="Training starten", command=self.start_training)
        self.start_btn.grid(row=3, column=0, columnspan=2, sticky="we", pady=(10, 0))

        self.progress = ttk.Progressbar(frame, mode="determinate", length=360)
        self.progress.grid(row=4, column=0, columnspan=2, sticky="we", pady=(10, 0))

        self.log_text = tk.Text(frame, width=70, height=8, state="disabled")
        self.log_text.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))

        # Live Plot Bereich
        self.fig = mpl_fig.Figure(figsize=(7, 3), dpi=100)
        self.ax_loss = self.fig.add_subplot(1, 2, 1)
        self.ax_acc = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout(pad=3.0)

        self._setup_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=7, column=0, columnspan=2, sticky="we", pady=(10, 0))

    def _setup_axes(self) -> None:
        self.ax_loss.clear()
        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylim(0, 2.5)

        self.ax_acc.clear()
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylim(0, 1.0)

    def _refresh_profile_label(self) -> None:
        profile = MODEL_PROFILES[self.size_var.get()]
        self.profile_label.config(
            text=(
                f"Profile: hidden={int(profile['hidden_size'])}, "
                f"layers={int(profile['hidden_layers'])}, "
                f"epochs={int(profile['epochs'])}, "
                f"batch={int(profile['batch_size'])}, "
                f"lr={float(profile['learning_rate'])}"
            )
        )

    def _append_log(self, message: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def start_training(self) -> None:
        if self.training_running:
            return

        version_raw = self.version_var.get().strip()
        try:
            version = validate_version(version_raw)
        except ValueError as exc:
            messagebox.showerror("Fehler", str(exc))
            return

        size = self.size_var.get().strip().lower()
        if size not in MODEL_PROFILES:
            messagebox.showerror("Fehler", "Ungueltige Modellgroesse.")
            return

        self.training_running = True
        self.start_btn.config(state="disabled")
        self.size_combo.config(state="disabled")
        self.version_entry.config(state="disabled")

        epochs = int(MODEL_PROFILES[size]["epochs"])
        self.progress["maximum"] = epochs
        self.progress["value"] = 0
        self.status_var.set("Training laeuft...")
        self._append_log(f"Starte Training: size={size}, version={version}, backend=gpu")

        # Reset Plot
        for key in self.history_data:
            self.history_data[key] = []
        self._setup_axes()
        self.canvas.draw()

        thread = threading.Thread(target=self._worker, args=(size, version), daemon=True)
        thread.start()
        self.root.after(150, self._poll_events)

    def _worker(self, size: str, version: str) -> None:
        def callback(event: str, data: dict[str, Any]) -> None:
            self.event_queue.put((event, data))

        try:
            metadata = train_model_gpu(
                size=size,
                version=version,
                platform_index=None,
                device_index=None,
                batch_size_override=None,
                test_eval_interval=1,
                fast_math=True,
                callback=callback,
            )
            self.event_queue.put(("success", {"metadata": metadata}))
        except BaseException as exc:  # noqa: BLE001
            details = traceback.format_exc()
            self.event_queue.put(("error", {"message": str(exc), "traceback": details}))

    def _poll_events(self) -> None:
        keep_polling = self.training_running

        while True:
            try:
                event, data = self.event_queue.get_nowait()
            except Empty:
                break

            if event == "info":
                message = str(data.get("message", ""))
                self._append_log(message)
                self.status_var.set(message)
            elif event == "start":
                self.progress["maximum"] = int(data.get("epochs", 1))
                self.progress["value"] = 0
            elif event == "progress":
                epoch = int(data.get("epoch", 0))
                epochs = int(data.get("epochs", 1))
                self.progress["value"] = epoch
                msg = (
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"Train Loss: {float(data['train_loss']):.4f} | Train Acc: {float(data['train_acc']):.4f} | "
                    f"Test Loss: {float(data['test_loss']):.4f} | Test Acc: {float(data['test_acc']):.4f}"
                )
                self._append_log(msg)
                self.status_var.set(f"Epoch {epoch}/{epochs}")

                # Update Plot
                self.history_data["epochs"].append(epoch)
                self.history_data["train_loss"].append(float(data["train_loss"]))
                self.history_data["test_loss"].append(float(data["test_loss"]))
                self.history_data["train_acc"].append(float(data["train_acc"]))
                self.history_data["test_acc"].append(float(data["test_acc"]))

                self.ax_loss.clear()
                self.ax_loss.set_title("Loss")
                self.ax_loss.plot(self.history_data["epochs"], self.history_data["train_loss"], "b-", label="Train")
                self.ax_loss.plot(self.history_data["epochs"], self.history_data["test_loss"], "r-", label="Test")
                self.ax_loss.legend(fontsize="small")
                self.ax_loss.set_ylim(0, max(2.5, max(self.history_data["train_loss"]) * 1.1 if self.history_data["train_loss"] else 2.5))

                self.ax_acc.clear()
                self.ax_acc.set_title("Accuracy")
                self.ax_acc.plot(self.history_data["epochs"], self.history_data["train_acc"], "b-", label="Train")
                self.ax_acc.plot(self.history_data["epochs"], self.history_data["test_acc"], "r-", label="Test")
                self.ax_acc.legend(fontsize="small", loc="lower right")
                self.ax_acc.set_ylim(0, 1.05)

                self.fig.tight_layout(pad=3.0)
                self.canvas.draw()
            elif event == "success":
                metadata = data.get("metadata", {})
                final_metrics = metadata.get("final_metrics", {}) if isinstance(metadata, dict) else {}
                final_acc = float(final_metrics.get("test_acc", 0.0))
                self._append_log(f"Finale Test-Accuracy: {final_acc:.4f}")
                self.status_var.set("Training abgeschlossen")
                self._finish_training()
                messagebox.showinfo("Fertig", f"Training abgeschlossen.\nFinale Test-Accuracy: {final_acc:.4f}")
                keep_polling = False
            elif event == "error":
                message = str(data.get("message", "Unbekannter Fehler"))
                trace = str(data.get("traceback", "")).strip()
                if trace:
                    self._append_log(trace)
                self._append_log("Fehler: " + message)
                self.status_var.set("Fehler")
                self._finish_training()
                messagebox.showerror("Training fehlgeschlagen", message)
                keep_polling = False

        if keep_polling:
            self.root.after(150, self._poll_events)

    def _finish_training(self) -> None:
        self.training_running = False
        self.start_btn.config(state="normal")
        self.size_combo.config(state="readonly")
        self.version_entry.config(state="normal")


def run_ui() -> None:
    root = tk.Tk()
    TrainingUI(root)
    root.mainloop()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        print_device_list()
        return

    if args.no_ui:
        if args.version is None:
            raise SystemExit("Im --no-ui Modus ist --version Pflicht.")
        run_cli(args)
        return

    run_ui()


if __name__ == "__main__":
    main()
