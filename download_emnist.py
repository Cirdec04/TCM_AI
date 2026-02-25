import gzip
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# Direkte Download-URL vom offiziellen NIST-Server
NIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


def download_with_progress(url: str, destination: Path) -> None:
    print(f"[1/4] Verbinde mit NIST-Server...")
    print(f"      URL: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    if total_size:
        print(f"      Dateigrösse: {total_size / (1024*1024):.1f} MB")
    else:
        print("      Dateigrösse unbekannt (Server sendet keine Content-Length).")

    print(f"[2/4] Downloade nach {destination.name}...")
    with open(destination, "wb") as f, tqdm(
        total=total_size if total_size else None,
        unit="B",
        unit_scale=True,
        desc="EMNIST Download",
    ) as progress:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))

    actual_size = destination.stat().st_size
    print(f"\n[2/4] Download fertig. Erhaltene Grösse: {actual_size / (1024*1024):.1f} MB")


def parse_idx(fd) -> np.ndarray:
    header = fd.read(4)
    if len(header) < 4:
        raise ValueError("Datei zu kurz um IDX-Header zu lesen.")
    dims = header[3]
    shape = tuple(int.from_bytes(fd.read(4), "big") for _ in range(dims))
    return np.frombuffer(fd.read(), dtype=np.uint8).reshape(shape)


def save_emnist_to_folders() -> None:
    base_dir = Path(__file__).resolve().parent
    temp_dir = base_dir / "temp_emnist"
    output_dir = base_dir / "data" / "EMNIST"
    zip_path = base_dir / "emnist_nist.zip"

    temp_dir.mkdir(exist_ok=True)

    # --- Schritt 1 & 2: Download ---
    if zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        if size_mb < 400:
            print(f"[1/4] Unvollständige Datei ({size_mb:.1f} MB). Lösche und lade neu herunter...")
            zip_path.unlink()
        else:
            print(f"[1/4] Vollständige ZIP bereits vorhanden ({size_mb:.1f} MB). Überspringe Download.")

    if not zip_path.exists():
        download_with_progress(NIST_URL, zip_path)

    # --- Schritt 3: Entpacken ---
    print(f"\n[3/4] Entpacke {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        digits_files = [f for f in all_files if "digits" in f]
        print(f"      Gesamt-Dateien im Archiv: {len(all_files)}")
        print(f"      Relevante Digits-Dateien: {len(digits_files)}")
        for name in digits_files:
            print(f"       - {name}")
        zf.extractall(temp_dir)
    print("      Entpacken fertig.")

    # --- Schritt 4: Bilder speichern ---
    data_map = {
        "training": (
            "emnist-digits-train-images-idx3-ubyte.gz",
            "emnist-digits-train-labels-idx1-ubyte.gz",
        ),
        "testing": (
            "emnist-digits-test-images-idx3-ubyte.gz",
            "emnist-digits-test-labels-idx1-ubyte.gz",
        ),
    }

    print(f"\n[4/4] Konvertiere Binärdaten zu PNG-Bildern...")
    for split, (img_file, lbl_file) in data_map.items():
        gzip_dir = temp_dir / "gzip"
        img_path = gzip_dir / img_file
        lbl_path = gzip_dir / lbl_file

        print(f"\n  --- {split.upper()} ---")
        print(f"      Lese Bilder aus: {img_file}")
        with gzip.open(img_path, "rb") as f:
            images = parse_idx(f)
            # EMNIST ist im Vergleich zu MNIST transponiert -> korrigieren
            images = images.swapaxes(1, 2)

        print(f"      Lese Labels aus: {lbl_file}")
        with gzip.open(lbl_path, "rb") as f:
            labels = parse_idx(f)

        print(f"      {len(images)} Samples gefunden. Speichere als PNG...")
        for i in tqdm(range(len(images)), desc=f"Speichere {split}"):
            label = int(labels[i])
            target_dir = output_dir / split / str(label)
            target_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(images[i], mode="L").save(
                target_dir / f"emnist_sample_{i}.png"
            )

    # --- Aufräumen ---
    print("\n[~] Räume temporäre Dateien auf...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("    Fertig!")

    print(f"\n✓ Alle EMNIST Digits wurden gespeichert nach: {output_dir}")
    print("  Ordnerstruktur: EMNIST/training/0..9/ und EMNIST/testing/0..9/")


if __name__ == "__main__":
    try:
        save_emnist_to_folders()
    except Exception as e:
        import traceback
        print(f"\n✗ Fehler: {e}")
        traceback.print_exc()
        input("\nDrücke Enter zum Schliessen...")
