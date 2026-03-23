from __future__ import annotations

import argparse
import json
import re
import threading
import tkinter as tk
import time
import traceback
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from tkinter import messagebox, ttk
from typing import Any, Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mpl_fig

from deps import ensure_requirements_installed

ensure_requirements_installed(required_modules=("numpy", "matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from nn import SimpleMLP


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "training"
TEST_DATA_DIR = DATA_DIR / "testing"
MODELS_DIR = BASE_DIR / "models"

MODEL_PROFILES = {
    "mini": {
        "hidden_sizes": [512, 256],
        "epochs": 96,
        "batch_size": 512,
    },
    "normal": {
        "hidden_sizes": [768, 384, 192],
        "epochs": 192,
        "batch_size": 512,
    },
    "pro": {
        "hidden_sizes": [2048, 1024, 512],
        "epochs": 512,
        "batch_size": 512,
    },
}
DEFAULT_ADAM_LEARNING_RATES = {
    "mini": 0.0015,
    "normal": 0.0015,
    "pro": 0.0008,
}

ProgressCallback = Callable[[str, dict[str, Any]], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiere ein einfaches MLP für Ziffernerkennung.")
    parser.add_argument("--no-ui", action="store_true", help="Kein GUI, direkt im Terminal trainieren.")
    parser.add_argument("--size", choices=["mini", "normal", "pro"], default="normal")
    parser.add_argument("--version", type=str, default=None, help="Versionsnummer des Modells (z. B. 2 oder 2.1).")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Stoppt, wenn sich die Test-Accuracy so viele Epochen nicht verbessert (0 = aus).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam Learning Rate (Standard: mini/normal=0.0015, pro=0.0008).",
    )
    parser.add_argument("--augment", action="store_true", help="Aktiviert Data Augmentation beim Training.")
    parser.add_argument(
        "--aug-prob",
        type=float,
        default=0.7,
        help="Wahrscheinlichkeit pro Sample fuer Augmentation (0.0 bis 1.0).",
    )
    parser.add_argument("--aug-shift", type=int, default=2, help="Maximaler Pixel-Shift in x/y Richtung.")
    parser.add_argument("--aug-rot", type=float, default=10.0, help="Maximaler Rotationswinkel in Grad.")
    return parser.parse_args()


def _emit(callback: ProgressCallback | None, event: str, **data: Any) -> None:
    if callback is not None:
        callback(event, data)


def _to_grayscale_unit(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32, copy=False)
    elif image.ndim == 3:
        channels = int(image.shape[2])
        if channels >= 3:
            rgb = image[..., :3].astype(np.float32, copy=False)
            gray = (0.299 * rgb[..., 0]) + (0.587 * rgb[..., 1]) + (0.114 * rgb[..., 2])
        else:
            gray = image[..., 0].astype(np.float32, copy=False)
    else:
        raise ValueError(f"Ungueltige Bildform: {image.shape}")

    if gray.size == 0:
        raise ValueError("Leeres Bild.")
    if float(np.max(gray)) > 1.0:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0).astype(np.float32, copy=False)


def _resize_nearest(image: np.ndarray, width: int = 28, height: int = 28) -> np.ndarray:
    if image.shape == (height, width):
        return image.astype(np.float32, copy=False)
    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    y_idx = np.linspace(0, src_h - 1, height).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, width).astype(np.int32)
    return image[np.ix_(y_idx, x_idx)].astype(np.float32, copy=False)


def _rotate_nearest_zero_fill(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if angle_deg == 0.0:
        return image
    h, w = image.shape
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    theta = np.deg2rad(angle_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    yy, xx = np.indices((h, w), dtype=np.float32)
    x0 = xx - cx
    y0 = yy - cy

    src_x = (c * x0) + (s * y0) + cx
    src_y = (-s * x0) + (c * y0) + cy

    src_x_i = np.rint(src_x).astype(np.int32)
    src_y_i = np.rint(src_y).astype(np.int32)

    valid = (src_x_i >= 0) & (src_x_i < w) & (src_y_i >= 0) & (src_y_i < h)
    out = np.zeros_like(image, dtype=np.float32)
    out[valid] = image[src_y_i[valid], src_x_i[valid]]
    return out


def _load_image_as_vector(path: Path) -> np.ndarray:
    pixels_raw = mpimg.imread(path)
    pixels = _to_grayscale_unit(np.asarray(pixels_raw))
    pixels = _resize_nearest(pixels, width=28, height=28)
    return pixels.reshape(-1)


def load_dataset_from_folders(
    data_dir: Path,
    callback: ProgressCallback | None = None,
    dataset_label: str = "Daten",
    progress_every: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
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

    if total_files == 0:
        raise ValueError(f"{dataset_label}: Keine Bilddateien gefunden in {data_dir}")

    x = np.empty((total_files, 28 * 28), dtype=np.float32)
    y = np.empty((total_files,), dtype=np.int64)

    for index, (digit_label, file_path) in enumerate(files_with_labels, start=1):
        x[index - 1] = _load_image_as_vector(file_path)
        y[index - 1] = digit_label
        if index == 1 or index % progress_every == 0 or index == total_files:
            _emit(callback, "info", message=f"{dataset_label}: {index}/{total_files} geladen.")

    return x, y


def count_images_in_dataset(data_dir: Path) -> int:
    total = 0
    for class_label in range(10):
        class_dir = data_dir / str(class_label)
        if not class_dir.exists():
            continue
        total += len([p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    return int(total)


def _shift_zero_fill(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(image)
    src_x0 = max(0, -dx)
    src_x1 = min(28, 28 - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(28, 28 - dy)
    dst_x0 = src_x0 + dx
    dst_x1 = src_x1 + dx
    dst_y0 = src_y0 + dy
    dst_y1 = src_y1 + dy
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return shifted


def augment_batch(
    x_batch: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.7,
    max_shift: int = 2,
    max_rotation: float = 10.0,
) -> np.ndarray:
    out = np.array(x_batch, dtype=np.float32, copy=True)
    if out.ndim != 2 or out.shape[1] != 28 * 28:
        raise ValueError("augment_batch erwartet Shape (batch, 784).")

    for sample_idx in range(out.shape[0]):
        if rng.random() > probability:
            continue

        image = out[sample_idx].reshape(28, 28)

        if max_shift > 0:
            dx = int(rng.integers(-max_shift, max_shift + 1))
            dy = int(rng.integers(-max_shift, max_shift + 1))
            image = _shift_zero_fill(image, dx=dx, dy=dy)

        if max_rotation > 0:
            angle = float(rng.uniform(-max_rotation, max_rotation))
            image = _rotate_nearest_zero_fill(image, angle_deg=angle)

        contrast = float(rng.uniform(0.85, 1.2))
        brightness = float(rng.uniform(-0.08, 0.08))
        image = np.clip((image * contrast) + brightness, 0.0, 1.0)

        if rng.random() < 0.35:
            noise_std = float(rng.uniform(0.01, 0.05))
            noise = rng.normal(0.0, noise_std, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        out[sample_idx] = image.reshape(-1)

    return out


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


def count_parameters(layer_sizes: list[int]) -> int:
    total = 0
    for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
        total += (in_features * out_features) + out_features
    return int(total)


def format_parameter_count(total: int) -> str:
    if total >= 1_000_000:
        return f"{total / 1_000_000:.2f}M"
    if total >= 1_000:
        return f"{total / 1_000:.1f}K"
    return str(total)


def cosine_learning_rate(base_lr: float, epoch: int, total_epochs: int, min_lr_ratio: float = 0.1) -> float:
    if total_epochs <= 1:
        return float(base_lr)
    progress = float(epoch - 1) / float(total_epochs - 1)
    cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
    return float(base_lr) * (float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine)


def compute_per_digit_accuracy(predictions: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> list[float]:
    preds = np.asarray(predictions, dtype=np.int64).reshape(-1)
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    per_digit: list[float] = []
    for digit in range(num_classes):
        mask = y_true == digit
        total = int(np.count_nonzero(mask))
        if total == 0:
            per_digit.append(0.0)
        else:
            correct = int(np.count_nonzero(preds[mask] == digit))
            per_digit.append(float(correct) / float(total))
    return per_digit


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


def train_model(
    size: str,
    version: str,
    callback: ProgressCallback | None = None,
    early_stopping_patience: int = 15,
    learning_rate_override: float | None = None,
    augment_enabled: bool = False,
    aug_prob: float = 0.7,
    aug_shift: int = 2,
    aug_rot: float = 10.0,
    stop_event: threading.Event | None = None,
) -> dict[str, object]:
    if size not in MODEL_PROFILES:
        raise ValueError("Ungueltige Groesse. Erlaubt: mini, normal, pro.")
    version = validate_version(version)
    if early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience muss >= 0 sein.")
    if learning_rate_override is not None and learning_rate_override <= 0:
        raise ValueError("--learning-rate muss > 0 sein.")
    if not 0.0 <= aug_prob <= 1.0:
        raise ValueError("--aug-prob muss zwischen 0.0 und 1.0 liegen.")
    if aug_shift < 0:
        raise ValueError("--aug-shift muss >= 0 sein.")
    if aug_rot < 0:
        raise ValueError("--aug-rot muss >= 0 sein.")

    train_data_dir = TRAIN_DATA_DIR
    test_data_dir = TEST_DATA_DIR
    models_dir = MODELS_DIR
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(f"Kein Schreibzugriff auf Modellordner: {models_dir}") from exc

    profile = MODEL_PROFILES[size]
    hidden_sizes = [int(x) for x in profile["hidden_sizes"]]
    hidden_layers = len(hidden_sizes)
    hidden_size = hidden_sizes[0]
    epochs = int(profile["epochs"])
    batch_size = int(profile["batch_size"])
    base_learning_rate = float(learning_rate_override) if learning_rate_override is not None else float(DEFAULT_ADAM_LEARNING_RATES[size])
    seed = get_fixed_seed()

    model_name = build_model_name(version=version, size=size)
    model_path = models_dir / f"{model_name}.npz"
    plot_path = models_dir / f"{model_name}_training.png"
    metadata_path = models_dir / f"{model_name}.json"

    if model_path.exists() or metadata_path.exists() or plot_path.exists():
        raise FileExistsError(
            f"Artefakt fuer '{model_name}' existiert bereits (.npz/.json/.png). Bitte andere Version waälen."
        )

    _emit(callback, "info", message=f"Lade Trainingsdaten aus: {train_data_dir}")
    x_train, y_train = load_dataset_from_folders(train_data_dir, callback=callback, dataset_label="Training")
    _emit(callback, "info", message=f"Geladene Trainings-Samples: {len(y_train)}")

    _emit(callback, "info", message=f"Lade Testdaten aus: {test_data_dir}")
    x_test, y_test = load_dataset_from_folders(test_data_dir, callback=callback, dataset_label="Testing")
    _emit(callback, "info", message=f"Geladene Test-Samples: {len(y_test)}")
    _emit(
        callback,
        "info",
        message=(
            "Training startet mit: "
            f"size={size}, hidden_sizes={hidden_sizes}, epochs={epochs}, "
            f"batch_size={batch_size}, optimizer=adam, learning_rate={base_learning_rate}, seed={seed}, "
            f"early_stopping_patience={early_stopping_patience}, "
            f"augmentation={augment_enabled} (prob={aug_prob}, shift={aug_shift}, rot={aug_rot})"
        ),
    )

    model = SimpleMLP(
        input_size=28 * 28,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        hidden_sizes=hidden_sizes,
        output_size=10,
        seed=seed,
    )
    _emit(callback, "info", message="Aktives Backend: CPU")

    xp = model.xp
    x_test_backend = model.asarray(x_test, dtype=xp.float32)
    y_test_backend = model.asarray(y_test, dtype=xp.int64)
    y_test_np = model.to_numpy(y_test_backend).astype(np.int64, copy=False)

    effective_batch_size = batch_size
    test_eval_interval = 1
    layer_sizes = [28 * 28] + hidden_sizes + [10]
    parameter_total = count_parameters(layer_sizes)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    history_test_acc_per_digit: list[list[float]] = []
    rng = np.random.default_rng(seed)
    early_stopping_enabled = early_stopping_patience > 0
    best_test_acc = float("-inf")
    best_test_epoch = 0
    best_weights_snapshot = [model.to_numpy(weight).copy() for weight in model.weights]
    best_biases_snapshot = [model.to_numpy(bias).copy() for bias in model.biases]
    last_test_acc_per_digit = [0.0] * 10
    epochs_without_improvement = 0
    stopped_early = False
    stopped_by_user = False

    _emit(callback, "start", epochs=epochs)

    start_time = time.perf_counter()
    num_samples = int(x_train.shape[0])
    num_batches = (num_samples + effective_batch_size - 1) // effective_batch_size
    progress_every = max(1, num_batches // 4)

    for epoch in range(1, epochs + 1):
        if stop_event is not None and stop_event.is_set():
            stopped_by_user = True
            break
        learning_rate = cosine_learning_rate(base_learning_rate, epoch=epoch, total_epochs=epochs)
        indices = rng.permutation(num_samples)

        batch_losses: list[float] = []
        batch_accs: list[float] = []

        for batch_idx, start in enumerate(range(0, num_samples, effective_batch_size), start=1):
            if stop_event is not None and stop_event.is_set():
                stopped_by_user = True
                break
            end = start + effective_batch_size
            batch_indices = indices[start:end]
            x_batch_np = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            if augment_enabled:
                x_batch_np = augment_batch(
                    x_batch_np,
                    rng=rng,
                    probability=float(aug_prob),
                    max_shift=int(aug_shift),
                    max_rotation=float(aug_rot),
                )
            x_batch = model.asarray(x_batch_np, dtype=xp.float32)
            y_batch_backend = model.asarray(y_batch, dtype=xp.int64)
            y_batch_one_hot = model.one_hot_labels(y_batch_backend, num_classes=10)
            batch_loss, batch_acc = model.train_batch(x_batch, y_batch_one_hot, learning_rate)
            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)
            if batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == num_batches:
                _emit(
                    callback,
                    "info",
                    message=f"Epoch {epoch}/{epochs}: Batch {batch_idx}/{num_batches}",
                )

        if stopped_by_user:
            break

        train_loss = float(np.mean(batch_losses))
        train_acc = float(np.mean(batch_accs))
        should_eval_test = (epoch == 1) or (epoch % test_eval_interval == 0) or (epoch == epochs)
        if should_eval_test:
            test_loss, test_acc = model.evaluate(x_test_backend, y_test_backend)
            test_preds = model.predict(x_test_backend)
            last_test_acc_per_digit = compute_per_digit_accuracy(test_preds, y_test_np, num_classes=10)
        else:
            test_loss = float(history["test_loss"][-1]) if history["test_loss"] else 0.0
            test_acc = float(history["test_acc"][-1]) if history["test_acc"] else 0.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history_test_acc_per_digit.append(list(last_test_acc_per_digit))

        _emit(
            callback,
            "progress",
            epoch=epoch,
            epochs=epochs,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            test_acc_per_digit=list(last_test_acc_per_digit),
        )

        if should_eval_test:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_epoch = epoch
                best_weights_snapshot = [model.to_numpy(weight).copy() for weight in model.weights]
                best_biases_snapshot = [model.to_numpy(bias).copy() for bias in model.biases]
                epochs_without_improvement = 0
            elif early_stopping_enabled:
                epochs_without_improvement += 1

            if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
                stopped_early = True
                _emit(
                    callback,
                    "info",
                    message=(
                        f"Early Stopping: keine Verbesserung der Test-Accuracy in "
                        f"{early_stopping_patience} Epochen. Stoppe bei Epoch {epoch}."
                    ),
                )
                break

    if stopped_by_user:
        _emit(callback, "info", message="Training manuell beendet (End Training Now).")

    training_time_seconds = time.perf_counter() - start_time
    epochs_trained = len(history["train_loss"])

    if history["test_acc"]:
        if best_test_epoch <= 0:
            best_idx = int(np.argmax(np.asarray(history["test_acc"], dtype=np.float32)))
            best_test_epoch = best_idx + 1
            best_test_acc = float(history["test_acc"][best_idx])
        best_metrics_idx = int(np.clip(best_test_epoch - 1, 0, len(history["test_acc"]) - 1))
        final_train_loss = float(history["train_loss"][best_metrics_idx])
        final_train_acc = float(history["train_acc"][best_metrics_idx])
        final_test_loss = float(history["test_loss"][best_metrics_idx])
        final_test_acc = float(history["test_acc"][best_metrics_idx])
        final_test_acc_per_digit = list(history_test_acc_per_digit[best_metrics_idx]) if history_test_acc_per_digit else [0.0] * 10
    else:
        best_test_acc = 0.0
        best_test_epoch = 0
        final_train_loss = 0.0
        final_train_acc = 0.0
        final_test_loss = 0.0
        final_test_acc = 0.0
        final_test_acc_per_digit = [0.0] * 10

    model.weights = [model.asarray(weight.astype(np.float32, copy=False), dtype=xp.float32) for weight in best_weights_snapshot]
    model.biases = [model.asarray(bias.astype(np.float32, copy=False), dtype=xp.float32) for bias in best_biases_snapshot]

    metadata: dict[str, object] = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_data_dir": str(train_data_dir),
        "test_data_dir": str(test_data_dir),
        "size": size,
        "hidden_sizes": hidden_sizes,
        "hidden_size": hidden_size,
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "epochs_trained": epochs_trained,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "test_eval_interval": test_eval_interval,
        "optimizer": "adam",
        "learning_rate": float(base_learning_rate),
        "parameters": {
            "total": int(parameter_total),
            "human": format_parameter_count(parameter_total),
        },
        "seed": seed,
        "compute_backend": "cpu",
        "augmentation": {
            "enabled": bool(augment_enabled),
            "probability": float(aug_prob),
            "max_shift": int(aug_shift),
            "max_rotation": float(aug_rot),
        },
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "patience": int(early_stopping_patience),
            "stopped_early": stopped_early,
            "best_test_acc": float(best_test_acc),
            "best_test_epoch": int(best_test_epoch),
        },
        "stopped_by_user": bool(stopped_by_user),
        "samples": {"total": int(len(y_train) + len(y_test)), "train": int(len(y_train)), "test": int(len(y_test))},
        "final_metrics": {
            "train_loss": final_train_loss,
            "train_acc": final_train_acc,
            "test_loss": final_test_loss,
            "test_acc": final_test_acc,
            "test_acc_per_digit": final_test_acc_per_digit,
        },
        "training_time_seconds": float(training_time_seconds),
        "artifacts": {
            "model_file": str(model_path),
            "plot_file": str(plot_path),
            "metadata_file": str(metadata_path),
        },
    }

    try:
        model.save(model_path, metadata=metadata)
        save_training_plot(history, plot_path)
        save_model_json(metadata, metadata_path)
    except PermissionError as exc:
        raise PermissionError(
            "Kein Schreibzugriff auf den Ordner 'models' oder auf eine dort gesperrte Datei. "
            "Bitte prüfe Dateirechte, schliesse geöffnete Dateien (z. B. Plot/JSON), und nutze eine neue Version."
        ) from exc

    _emit(callback, "info", message="Training fertig.")
    _emit(callback, "info", message=f"Modell gespeichert: {model_path}")
    _emit(callback, "info", message=f"Plot gespeichert:   {plot_path}")
    _emit(callback, "info", message=f"JSON gespeichert:   {metadata_path}")

    return metadata


def run_cli(
    size: str,
    version: str,
    early_stopping_patience: int,
    learning_rate: float | None,
    augment_enabled: bool,
    aug_prob: float,
    aug_shift: int,
    aug_rot: float,
) -> None:
    def callback(event: str, data: dict[str, Any]) -> None:
        if event == "progress":
            print(
                f"Epoch {data['epoch']:02d}/{data['epochs']} | "
                f"Train Loss: {data['train_loss']:.4f} | Train Acc: {data['train_acc']:.4f} | "
                f"Test Loss: {data['test_loss']:.4f} | Test Acc: {data['test_acc']:.4f}"
            )
        elif event == "info":
            print(data["message"])

    metadata = train_model(
        size=size,
        version=version,
        callback=callback,
        early_stopping_patience=early_stopping_patience,
        learning_rate_override=learning_rate,
        augment_enabled=augment_enabled,
        aug_prob=aug_prob,
        aug_shift=aug_shift,
        aug_rot=aug_rot,
    )
    final_acc = float((metadata.get("final_metrics", {}) or {}).get("test_acc", 0.0))
    print(f"Finale Test-Accuracy: {final_acc:.4f}")


class TrainingUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Training Konfiguration (CPU)")
        self.root.resizable(False, False)

        self.event_queue: Queue[tuple[str, dict[str, Any]]] = Queue()
        self.training_running = False
        self.stop_training_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.size_var = tk.StringVar(value="normal")
        self.version_var = tk.StringVar(value="1")
        self.augment_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Bereit")
        self.live_epoch_var = tk.StringVar(value="-")
        self.live_train_loss_var = tk.StringVar(value="-")
        self.live_train_acc_var = tk.StringVar(value="-")
        self.live_test_loss_var = tk.StringVar(value="-")
        self.live_test_acc_var = tk.StringVar(value="-")
        self.live_test_acc_per_digit_var = tk.StringVar(value="-")

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

        self.augment_check = ttk.Checkbutton(frame, text="Data Augmentation", variable=self.augment_var)
        self.augment_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.profile_label = ttk.Label(frame, text="", justify="left")
        self.profile_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.start_btn = ttk.Button(frame, text="Training starten", command=self.start_training)
        self.start_btn.grid(row=5, column=0, sticky="we", pady=(10, 0), padx=(0, 4))
        self.stop_btn = ttk.Button(frame, text="End Training Now", command=self.stop_training, state="disabled")
        self.stop_btn.grid(row=5, column=1, sticky="we", pady=(10, 0), padx=(4, 0))

        self.progress = ttk.Progressbar(frame, mode="determinate", length=360)
        self.progress.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))

        metrics_frame = ttk.LabelFrame(frame, text="Live Metriken", padding=8)
        metrics_frame.grid(row=7, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(metrics_frame, text="Epoch:").grid(row=0, column=0, sticky="w")
        ttk.Label(metrics_frame, textvariable=self.live_epoch_var).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(metrics_frame, text="Train Loss:").grid(row=1, column=0, sticky="w")
        ttk.Label(metrics_frame, textvariable=self.live_train_loss_var).grid(row=1, column=1, sticky="w", padx=(8, 0))
        ttk.Label(metrics_frame, text="Train Acc:").grid(row=2, column=0, sticky="w")
        ttk.Label(metrics_frame, textvariable=self.live_train_acc_var).grid(row=2, column=1, sticky="w", padx=(8, 0))
        ttk.Label(metrics_frame, text="Test Loss:").grid(row=3, column=0, sticky="w")
        ttk.Label(metrics_frame, textvariable=self.live_test_loss_var).grid(row=3, column=1, sticky="w", padx=(8, 0))
        ttk.Label(metrics_frame, text="Test Acc:").grid(row=4, column=0, sticky="w")
        ttk.Label(metrics_frame, textvariable=self.live_test_acc_var).grid(row=4, column=1, sticky="w", padx=(8, 0))
        ttk.Label(metrics_frame, text="Test Acc 0-9:").grid(row=5, column=0, sticky="nw")
        ttk.Label(metrics_frame, textvariable=self.live_test_acc_per_digit_var, justify="left").grid(row=5, column=1, sticky="w", padx=(8, 0))

        self.log_text = tk.Text(frame, width=70, height=8, state="disabled")
        self.log_text.grid(row=8, column=0, columnspan=2, sticky="we", pady=(10, 0))

    def _refresh_profile_label(self) -> None:
        profile = MODEL_PROFILES[self.size_var.get()]
        hidden_sizes = profile["hidden_sizes"]
        self.profile_label.config(
            text=(
                f"Profile: hidden_sizes={hidden_sizes}, "
                f"epochs={int(profile['epochs'])}, "
                f"batch={int(profile['batch_size'])}, "
                "optimizer=adam"
            )
        )

    def _format_per_digit_accuracy(self, values: list[float]) -> str:
        first = "  ".join([f"{digit}:{(float(values[digit]) * 100.0):5.1f}%" for digit in range(5)])
        second = "  ".join([f"{digit}:{(float(values[digit]) * 100.0):5.1f}%" for digit in range(5, 10)])
        return first + "\n" + second

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
        self.stop_training_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.size_combo.config(state="disabled")
        self.version_entry.config(state="disabled")
        self.augment_check.config(state="disabled")

        epochs = int(MODEL_PROFILES[size]["epochs"])
        self.progress["maximum"] = epochs
        self.progress["value"] = 0
        self.status_var.set("Training laeuft...")
        augment_enabled = bool(self.augment_var.get())
        self._append_log(
            f"Starte Training: size={size}, version={version}, augment={augment_enabled}, backend=cpu"
        )

        self.worker_thread = threading.Thread(
            target=self._worker,
            args=(size, version, augment_enabled),
            daemon=False,
        )
        self.worker_thread.start()
        self.root.after(150, self._poll_events)

    def _on_close(self) -> None:
        if self.training_running:
            self.stop_training_event.set()
            self.stop_btn.config(state="disabled")
            self.status_var.set("Stop angefordert... Fenster schliesst nach Trainingsende.")
            self._append_log("Fenster schliessen: Stop angefordert, warte auf sicheren Abbruch...")
            self.root.after(150, self._wait_for_shutdown)
            return
        self.root.destroy()

    def _wait_for_shutdown(self) -> None:
        if self.training_running:
            self.root.after(150, self._wait_for_shutdown)
            return
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.root.after(150, self._wait_for_shutdown)
            return
        self.root.destroy()

    def stop_training(self) -> None:
        if not self.training_running:
            return
        self.stop_training_event.set()
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stop angefordert...")
        self._append_log("Manueller Stopp angefordert (End Training Now). Warte auf sicheren Abbruch...")

    def _worker(self, size: str, version: str, augment_enabled: bool) -> None:
        def callback(event: str, data: dict[str, Any]) -> None:
            self.event_queue.put((event, data))

        try:
            metadata = train_model(
                size=size,
                version=version,
                callback=callback,
                early_stopping_patience=15,
                augment_enabled=augment_enabled,
                stop_event=self.stop_training_event,
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
                train_loss = float(data["train_loss"])
                train_acc = float(data["train_acc"])
                test_loss = float(data["test_loss"])
                test_acc = float(data["test_acc"])
                per_digit = [float(v) for v in data.get("test_acc_per_digit", [0.0] * 10)]
                self.progress["value"] = epoch
                msg = (
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
                )
                self._append_log(msg)
                self.status_var.set(f"Epoch {epoch}/{epochs}")
                self.live_epoch_var.set(f"{epoch}/{epochs}")
                self.live_train_loss_var.set(f"{train_loss:.4f}")
                self.live_train_acc_var.set(f"{train_acc:.4f}")
                self.live_test_loss_var.set(f"{test_loss:.4f}")
                self.live_test_acc_var.set(f"{test_acc:.4f}")
                self.live_test_acc_per_digit_var.set(self._format_per_digit_accuracy(per_digit))
            elif event == "success":
                metadata = data.get("metadata", {})
                final_metrics = metadata.get("final_metrics", {}) if isinstance(metadata, dict) else {}
                stopped_by_user = bool(metadata.get("stopped_by_user", False)) if isinstance(metadata, dict) else False
                final_acc = float(final_metrics.get("test_acc", 0.0))
                self._append_log(f"Finale Test-Accuracy: {final_acc:.4f}")
                self.status_var.set("Training manuell beendet" if stopped_by_user else "Training abgeschlossen")
                self._finish_training()
                if stopped_by_user:
                    messagebox.showinfo("Beendet", f"Training manuell beendet.\nFinale Test-Accuracy: {final_acc:.4f}")
                else:
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
        self.stop_btn.config(state="disabled")
        self.size_combo.config(state="readonly")
        self.version_entry.config(state="normal")
        self.augment_check.config(state="normal")
        if self.worker_thread is not None and not self.worker_thread.is_alive():
            self.worker_thread = None


def run_ui() -> None:
    root = tk.Tk()
    TrainingUI(root)
    root.mainloop()


def main() -> None:
    args = parse_args()

    if args.no_ui:
        if args.version is None:
            raise SystemExit("Im --no-ui Modus ist --version Pflicht.")
        run_cli(
            size=args.size,
            version=args.version,
            early_stopping_patience=args.early_stopping_patience,
            learning_rate=args.learning_rate,
            augment_enabled=args.augment,
            aug_prob=args.aug_prob,
            aug_shift=args.aug_shift,
            aug_rot=args.aug_rot,
        )
        return

    run_ui()


if __name__ == "__main__":
    main()

