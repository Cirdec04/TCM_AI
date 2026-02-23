from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit("Fehlende Abhaengigkeit: Pillow. Bitte `pip install -r requirements.txt` ausfuehren.") from exc

from nn import SimpleMLP, one_hot


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SIZE_TO_HIDDEN = {
    "mini": 64,
    "normal": 128,
    "pro": 256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiere ein einfaches MLP fuer Ziffernerkennung.")
    parser.add_argument("--data-dir", default="data", help="Pfad zum Datenordner mit Klassen 0-9.")
    parser.add_argument("--models-dir", default="models", help="Pfad zum Modellordner.")
    parser.add_argument("--size", choices=["mini", "normal", "pro"], default="normal")
    parser.add_argument("--version", type=int, default=None, help="Versionsnummer fuer Naming-Schema (Pflicht ohne Prompt).")
    parser.add_argument("--hidden-size", type=int, default=None, help="Optional: ueberschreibt hidden size.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Optionales Limit fuer schnelle Tests.")
    parser.add_argument("--no-prompt", action="store_true", help="Kein Fragen-Modus, nur CLI-Parameter.")
    return parser.parse_args()


def _load_image_as_vector(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((28, 28))
    pixels = np.asarray(image, dtype=np.float32)

    # Falls Hintergrund hell ist, invertieren wir auf "MNIST-Stil" (helle Ziffer auf dunklem Hintergrund).
    if float(pixels.mean()) > 127.0:
        pixels = 255.0 - pixels

    pixels = pixels / 255.0
    return pixels.reshape(-1)


def load_dataset_from_folders(data_dir: Path, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
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
            if max_samples is not None and len(features) >= max_samples:
                break
        if max_samples is not None and len(features) >= max_samples:
            break

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


def _prompt_float(
    question: str,
    default: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    while True:
        text = _prompt_str(question, None if default is None else str(default))
        try:
            value = float(text)
        except ValueError:
            print("Bitte eine Zahl eingeben.")
            continue
        if min_value is not None and value <= min_value:
            print(f"Bitte eine Zahl > {min_value} eingeben.")
            continue
        if max_value is not None and value >= max_value:
            print(f"Bitte eine Zahl < {max_value} eingeben.")
            continue
        return value


def _prompt_optional_int(question: str, default: int | None = None) -> int | None:
    default_text = "none" if default is None else str(default)
    while True:
        text = _prompt_str(question + " (Zahl oder 'none')", default_text).lower()
        if text in {"none", "null", "kein", "keine", ""}:
            return None
        try:
            value = int(text)
        except ValueError:
            print("Bitte Zahl oder 'none' eingeben.")
            continue
        if value <= 0:
            print("Bitte eine Zahl > 0 eingeben.")
            continue
        return value


def prompt_for_training_settings(args: argparse.Namespace, models_dir: Path) -> argparse.Namespace:
    print("\n=== Training Konfiguration ===")
    print("Einfach Enter druecken, um den Standardwert zu behalten.\n")

    args.data_dir = _prompt_str("Datenordner", args.data_dir)
    args.models_dir = _prompt_str("Modellordner", args.models_dir)

    while True:
        size = _prompt_str("Modellgroesse (mini/normal/pro)", args.size).lower()
        if size in SIZE_TO_HIDDEN:
            args.size = size
            break
        print("Ungueltig. Erlaubt: mini, normal, pro.")

    while True:
        version = _prompt_int("Version (Pflicht, z. B. 1, 2, 3)", args.version, min_value=1)
        candidate_name = build_model_name(version, args.size)
        if (Path(args.models_dir) / f"{candidate_name}.npz").exists():
            print(f"Modell '{candidate_name}.npz' existiert bereits. Bitte andere Version waehlen.")
            args.version = None
            continue
        args.version = version
        break

    default_hidden = args.hidden_size if args.hidden_size is not None else SIZE_TO_HIDDEN[args.size]
    args.hidden_size = _prompt_int("Hidden Size", default_hidden, min_value=1)
    args.epochs = _prompt_int("Epochen", args.epochs, min_value=1)
    args.batch_size = _prompt_int("Batch Size", args.batch_size, min_value=1)
    args.learning_rate = _prompt_float("Learning Rate", args.learning_rate, min_value=0.0)
    args.test_ratio = _prompt_float("Test Ratio (z. B. 0.2)", args.test_ratio, min_value=0.0, max_value=1.0)
    args.seed = _prompt_int("Seed", args.seed, min_value=0)
    args.max_samples = _prompt_optional_int("Max Samples", args.max_samples)
    return args


def save_training_plot(history: dict[str, list[float]], output_path: Path) -> bool:
    if plt is None:
        return False

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
    return True


def main() -> None:
    args = parse_args()

    if args.no_prompt:
        if args.version is None:
            raise SystemExit("Ohne Prompt ist --version Pflicht. Oder starte ohne --no-prompt.")
        if args.hidden_size is None:
            args.hidden_size = SIZE_TO_HIDDEN[args.size]
    else:
        args = prompt_for_training_settings(args, Path(args.models_dir))

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lade Daten aus: {data_dir}")
    x, y = load_dataset_from_folders(data_dir, max_samples=args.max_samples)
    print(f"Geladene Samples: {len(y)}")

    x_train, y_train, x_test, y_test = split_train_test(x, y, args.test_ratio, args.seed)
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")

    hidden_size = args.hidden_size if args.hidden_size is not None else SIZE_TO_HIDDEN[args.size]
    model = SimpleMLP(input_size=28 * 28, hidden_size=hidden_size, output_size=10, seed=args.seed)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        indices = np.arange(x_train.shape[0])
        rng.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        batch_losses: list[float] = []
        batch_accs: list[float] = []

        for start in range(0, x_train_shuffled.shape[0], args.batch_size):
            end = start + args.batch_size
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            y_batch_one_hot = one_hot(y_batch, num_classes=10)
            batch_loss, batch_acc = model.train_batch(x_batch, y_batch_one_hot, args.learning_rate)
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
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    if args.version is None:
        raise SystemExit("Version fehlt. Bitte Version eingeben.")
    version = args.version
    model_name = build_model_name(version=version, size=args.size)
    model_path = models_dir / f"{model_name}.npz"
    plot_path = models_dir / f"{model_name}_training.png"
    if model_path.exists():
        raise FileExistsError(
            f"Modell existiert bereits: {model_path}. "
            "Bitte --version erhoehen oder Datei umbenennen/loeschen."
        )

    metadata = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "size": args.size,
        "hidden_size": hidden_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }

    model.save(model_path, metadata=metadata)
    plot_written = save_training_plot(history, plot_path)

    print("\nTraining fertig.")
    print(f"Modell gespeichert: {model_path}")
    if plot_written:
        print(f"Plot gespeichert:   {plot_path}")
    else:
        print("Plot uebersprungen: matplotlib nicht installiert.")
    print(f"Finale Test-Accuracy: {history['test_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
