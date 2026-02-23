from __future__ import annotations

import argparse
import json
import secrets
from datetime import datetime
from pathlib import Path

from deps import ensure_requirements_installed

ensure_requirements_installed(required_modules=("numpy", "matplotlib", "PIL"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nn import SimpleMLP, one_hot


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
TEST_RATIO = 0.2

MODEL_PROFILES: dict[str, dict[str, float | int]] = {
    "mini": {
        "hidden_size": 64,
        "epochs": 12,
        "batch_size": 64,
        "learning_rate": 0.01,
    },
    "normal": {
        "hidden_size": 128,
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.008,
    },
    "pro": {
        "hidden_size": 256,
        "epochs": 35,
        "batch_size": 128,
        "learning_rate": 0.006,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiere ein einfaches MLP fuer Ziffernerkennung.")
    parser.add_argument("--size", choices=["mini", "normal", "pro"], default="normal")
    parser.add_argument("--version", type=int, default=None, help="Versionsnummer fuer Naming-Schema (Pflicht ohne Prompt).")
    parser.add_argument("--no-prompt", action="store_true", help="Kein Fragen-Modus, nur CLI-Parameter.")
    return parser.parse_args()


def _prompt_str(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{question}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("Bitte einen Wert eingeben.")


def _prompt_int(question: str, default: int | None = None, min_value: int | None = None) -> int:
    while True:
        text = _prompt_str(question, None if default is None else str(default))
        try:
            value = int(text)
        except ValueError:
            print("Bitte eine ganze Zahl eingeben.")
            continue
        if min_value is not None and value < min_value:
            print(f"Bitte eine Zahl >= {min_value} eingeben.")
            continue
        return value


def _load_image_as_vector(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((28, 28))
    pixels = np.asarray(image, dtype=np.float32)

    # Falls Hintergrund hell ist, invertieren wir auf "MNIST-Stil" (helle Ziffer auf dunklem Hintergrund).
    if float(pixels.mean()) > 127.0:
        pixels = 255.0 - pixels

    pixels = pixels / 255.0
    return pixels.reshape(-1)


def load_dataset_from_folders(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    for label in range(10):
        class_dir = data_dir / str(label)
        if not class_dir.exists():
            continue

        files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        for file_path in files:
            features.append(_load_image_as_vector(file_path))
            labels.append(label)

    if not features:
        raise FileNotFoundError(
            f"Keine Trainingsdaten gefunden in '{data_dir}'. "
            "Erwartet werden Unterordner 0 bis 9 mit Bilddateien."
        )

    x = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    test_size = int(len(indices) * test_ratio)
    test_size = max(1, test_size)
    test_size = min(len(indices) - 1, test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]
    return x_train, y_train, x_test, y_test


def build_model_name(version: int, size: str) -> str:
    suffix = "" if size == "normal" else f"-{size}"
    return f"TCM-o{version}{suffix}"


def generate_random_seed() -> int:
    # 32-bit positiver Bereich fuer reproduzierbare NumPy-Nutzung.
    return secrets.randbelow(2_147_483_647) + 1


def prompt_for_training_settings(args: argparse.Namespace, models_dir: Path) -> argparse.Namespace:
    print("\n=== Training Konfiguration ===")
    print("Einfach Enter druecken, um den Standardwert zu behalten.\n")

    while True:
        size = _prompt_str("Modellgroesse (mini/normal/pro)", args.size).lower()
        if size in MODEL_PROFILES:
            args.size = size
            break
        print("Ungueltig. Erlaubt: mini, normal, pro.")

    profile = MODEL_PROFILES[args.size]
    print(
        "Aktive Standards: "
        f"hidden_size={int(profile['hidden_size'])}, "
        f"epochs={int(profile['epochs'])}, "
        f"batch_size={int(profile['batch_size'])}, "
        f"learning_rate={float(profile['learning_rate'])}"
    )

    while True:
        version = _prompt_int("Version (Pflicht, z. B. 1, 2, 3)", args.version, min_value=1)
        candidate_name = build_model_name(version, args.size)
        if (models_dir / f"{candidate_name}.npz").exists() or (models_dir / f"{candidate_name}.json").exists():
            print(f"Modell '{candidate_name}' existiert bereits. Bitte andere Version waehlen.")
            args.version = None
            continue
        args.version = version
        break

    return args


def save_training_plot(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Verlauf")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Verlauf")
    plt.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()


def save_model_json(metadata: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.no_prompt:
        if args.version is None:
            raise SystemExit("Ohne Prompt ist --version Pflicht. Oder starte ohne --no-prompt.")
        if args.size not in MODEL_PROFILES:
            raise SystemExit("Ungueltige Groesse. Erlaubt: mini, normal, pro.")
    else:
        args = prompt_for_training_settings(args, models_dir)

    if args.version is None:
        raise SystemExit("Version fehlt. Bitte Version eingeben.")

    profile = MODEL_PROFILES[args.size]
    hidden_size = int(profile["hidden_size"])
    epochs = int(profile["epochs"])
    batch_size = int(profile["batch_size"])
    learning_rate = float(profile["learning_rate"])

    seed = generate_random_seed()

    model_name = build_model_name(version=args.version, size=args.size)
    model_path = models_dir / f"{model_name}.npz"
    plot_path = models_dir / f"{model_name}_training.png"
    metadata_path = models_dir / f"{model_name}.json"

    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Modell existiert bereits: {model_name}. "
            "Bitte Version erhoehen oder Datei umbenennen/loeschen."
        )

    print(f"\nLade Daten aus: {data_dir}")
    x, y = load_dataset_from_folders(data_dir)
    print(f"Geladene Samples: {len(y)}")

    x_train, y_train, x_test, y_test = split_train_test(x, y, TEST_RATIO, seed)
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    print(
        "Training startet mit: "
        f"size={args.size}, hidden_size={hidden_size}, epochs={epochs}, "
        f"batch_size={batch_size}, learning_rate={learning_rate}, seed={seed}"
    )

    model = SimpleMLP(input_size=28 * 28, hidden_size=hidden_size, output_size=10, seed=seed)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        indices = np.arange(x_train.shape[0])
        rng.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        batch_losses: list[float] = []
        batch_accs: list[float] = []

        for start in range(0, x_train_shuffled.shape[0], batch_size):
            end = start + batch_size
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            y_batch_one_hot = one_hot(y_batch, num_classes=10)
            batch_loss, batch_acc = model.train_batch(x_batch, y_batch_one_hot, learning_rate)
            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)

        train_loss = float(np.mean(batch_losses))
        train_acc = float(np.mean(batch_accs))
        test_loss, test_acc = model.evaluate(x_test, y_test)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    metadata = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "size": args.size,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "test_ratio": TEST_RATIO,
        "seed": seed,
        "samples": {
            "total": int(len(y)),
            "train": int(len(y_train)),
            "test": int(len(y_test)),
        },
        "final_metrics": {
            "train_loss": float(history["train_loss"][-1]),
            "train_acc": float(history["train_acc"][-1]),
            "test_loss": float(history["test_loss"][-1]),
            "test_acc": float(history["test_acc"][-1]),
        },
        "artifacts": {
            "model_file": str(model_path),
            "plot_file": str(plot_path),
            "metadata_file": str(metadata_path),
        },
    }

    model.save(model_path, metadata=metadata)
    save_training_plot(history, plot_path)
    save_model_json(metadata, metadata_path)

    print("\nTraining fertig.")
    print(f"Modell gespeichert: {model_path}")
    print(f"Plot gespeichert:   {plot_path}")
    print(f"JSON gespeichert:   {metadata_path}")
    print(f"Finale Test-Accuracy: {history['test_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
