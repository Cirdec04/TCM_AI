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

from deps import ensure_requirements_installed

ensure_requirements_installed(required_modules=("numpy", "matplotlib", "PIL"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nn import SimpleMLP


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
        "batch_size": 128,
        "learning_rate": 0.0025,
    },
    "normal": {
        "hidden_size": 512,
        "hidden_layers": 2,
        "epochs": 192,
        "batch_size": 128,
        "learning_rate": 0.0015,
    },
    "pro": {
        "hidden_size": 2048,
        "hidden_layers": 3,
        "epochs": 512,
        "batch_size": 128,
        "learning_rate": 0.001,
    },
}

ProgressCallback = Callable[[str, dict[str, Any]], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiere ein einfaches MLP fuer Ziffernerkennung.")
    parser.add_argument("--no-ui", action="store_true", help="Kein GUI, direkt im Terminal trainieren.")
    parser.add_argument("--size", choices=["mini", "normal", "pro"], default="normal")
    parser.add_argument("--version", type=str, default=None, help="Versionsnummer des Modells (z. B. 2 oder 2.1).")
    return parser.parse_args()


def _emit(callback: ProgressCallback | None, event: str, **data: Any) -> None:
    if callback is not None:
        callback(event, data)


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
    callback: ProgressCallback | None = None,
    dataset_label: str = "Daten",
    progress_every: int = 500,
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

    for index, (digit_label, file_path) in enumerate(files_with_labels, start=1):
        features.append(_load_image_as_vector(file_path))
        labels.append(digit_label)

        if index == 1 or index % progress_every == 0 or index == total_files:
            _emit(callback, "info", message=f"{dataset_label}: {index}/{total_files} geladen.")

    x = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
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


def train_model(size: str, version: str, callback: ProgressCallback | None = None) -> dict[str, object]:
    if size not in MODEL_PROFILES:
        raise ValueError("Ungueltige Groesse. Erlaubt: mini, normal, pro.")
    version = validate_version(version)

    train_data_dir = TRAIN_DATA_DIR
    test_data_dir = TEST_DATA_DIR
    models_dir = MODELS_DIR
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(f"Kein Schreibzugriff auf Modellordner: {models_dir}") from exc

    profile = MODEL_PROFILES[size]
    hidden_size = int(profile["hidden_size"])
    hidden_layers = int(profile["hidden_layers"])
    epochs = int(profile["epochs"])
    batch_size = int(profile["batch_size"])
    learning_rate = float(profile["learning_rate"])
    seed = get_fixed_seed()

    model_name = build_model_name(version=version, size=size)
    model_path = models_dir / f"{model_name}.npz"
    plot_path = models_dir / f"{model_name}_training.png"
    metadata_path = models_dir / f"{model_name}.json"

    if model_path.exists() or metadata_path.exists() or plot_path.exists():
        raise FileExistsError(
            f"Artefakt fuer '{model_name}' existiert bereits (.npz/.json/.png). Bitte andere Version waehlen."
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
            f"size={size}, hidden_size={hidden_size}, hidden_layers={hidden_layers}, epochs={epochs}, "
            f"batch_size={batch_size}, learning_rate={learning_rate}, seed={seed}"
        ),
    )

    model = SimpleMLP(
        input_size=28 * 28,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        output_size=10,
        seed=seed,
    )
    _emit(callback, "info", message="Aktives Backend: CPU")

    xp = model.xp
    x_train_backend = model.asarray(x_train, dtype=xp.float32)
    y_train_backend = model.asarray(y_train, dtype=xp.int64)
    x_test_backend = model.asarray(x_test, dtype=xp.float32)
    y_test_backend = model.asarray(y_test, dtype=xp.int64)

    effective_batch_size = batch_size
    test_eval_interval = 1

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    rng = np.random.default_rng(seed)

    _emit(callback, "start", epochs=epochs)

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        indices = np.arange(x_train.shape[0])
        rng.shuffle(indices)
        x_train_shuffled = x_train_backend[indices]
        y_train_shuffled = y_train_backend[indices]

        batch_losses: list[float] = []
        batch_accs: list[float] = []

        num_samples = int(x_train_shuffled.shape[0])
        num_batches = (num_samples + effective_batch_size - 1) // effective_batch_size
        progress_every = max(1, num_batches // 4)

        for batch_idx, start in enumerate(range(0, num_samples, effective_batch_size), start=1):
            end = start + effective_batch_size
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            y_batch_one_hot = model.one_hot_labels(y_batch, num_classes=10)
            batch_loss, batch_acc = model.train_batch(x_batch, y_batch_one_hot, learning_rate)
            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)
            if batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == num_batches:
                _emit(
                    callback,
                    "info",
                    message=f"Epoch {epoch}/{epochs}: Batch {batch_idx}/{num_batches}",
                )

        train_loss = float(np.mean(batch_losses))
        train_acc = float(np.mean(batch_accs))
        should_eval_test = (epoch == 1) or (epoch % test_eval_interval == 0) or (epoch == epochs)
        if should_eval_test:
            test_loss, test_acc = model.evaluate(x_test_backend, y_test_backend)
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

    training_time_seconds = time.perf_counter() - start_time

    metadata: dict[str, object] = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_data_dir": str(train_data_dir),
        "test_data_dir": str(test_data_dir),
        "size": size,
        "hidden_size": hidden_size,
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "test_eval_interval": test_eval_interval,
        "learning_rate": learning_rate,
        "seed": seed,
        "compute_backend": "cpu",
        "samples": {"total": int(len(y_train) + len(y_test)), "train": int(len(y_train)), "test": int(len(y_test))},
        "final_metrics": {
            "train_loss": float(history["train_loss"][-1]),
            "train_acc": float(history["train_acc"][-1]),
            "test_loss": float(history["test_loss"][-1]),
            "test_acc": float(history["test_acc"][-1]),
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
            "Bitte pruefe Dateirechte, schliesse geoeffnete Dateien (z. B. Plot/JSON), und nutze eine neue Version."
        ) from exc

    _emit(callback, "info", message="Training fertig.")
    _emit(callback, "info", message=f"Modell gespeichert: {model_path}")
    _emit(callback, "info", message=f"Plot gespeichert:   {plot_path}")
    _emit(callback, "info", message=f"JSON gespeichert:   {metadata_path}")

    return metadata


def run_cli(size: str, version: str) -> None:
    def callback(event: str, data: dict[str, Any]) -> None:
        if event == "progress":
            print(
                f"Epoch {data['epoch']:02d}/{data['epochs']} | "
                f"Train Loss: {data['train_loss']:.4f} | Train Acc: {data['train_acc']:.4f} | "
                f"Test Loss: {data['test_loss']:.4f} | Test Acc: {data['test_acc']:.4f}"
            )
        elif event == "info":
            print(data["message"])

    metadata = train_model(size=size, version=version, callback=callback)
    final_acc = float((metadata.get("final_metrics", {}) or {}).get("test_acc", 0.0))
    print(f"Finale Test-Accuracy: {final_acc:.4f}")


class TrainingUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Training Konfiguration")
        self.root.resizable(False, False)

        self.event_queue: Queue[tuple[str, dict[str, Any]]] = Queue()
        self.training_running = False

        self.size_var = tk.StringVar(value="normal")
        self.version_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Bereit")

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

        ttk.Label(frame, textvariable=self.status_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.log_text = tk.Text(frame, width=70, height=12, state="disabled")
        self.log_text.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))

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
        self._append_log(f"Starte Training: size={size}, version={version}, backend=cpu")

        thread = threading.Thread(target=self._worker, args=(size, version), daemon=True)
        thread.start()
        self.root.after(150, self._poll_events)

    def _worker(self, size: str, version: str) -> None:
        def callback(event: str, data: dict[str, Any]) -> None:
            self.event_queue.put((event, data))

        try:
            metadata = train_model(size=size, version=version, callback=callback)
            self.event_queue.put(("success", {"metadata": metadata}))
        except Exception as exc:  # noqa: BLE001
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

    if args.no_ui:
        if args.version is None:
            raise SystemExit("Im --no-ui Modus ist --version Pflicht.")
        run_cli(size=args.size, version=args.version)
        return

    run_ui()


if __name__ == "__main__":
    main()
