import os
import subprocess
import sys
from pathlib import Path

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installiere {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Benötigte Pakete sicherstellen
install_and_import("emnist")
install_and_import("tqdm")
install_and_import("PIL")
install_and_import("numpy")

import numpy as np
from emnist import extract_training_samples, extract_test_samples
from PIL import Image
from tqdm import tqdm

def save_emnist_to_folders():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "data" / "emnist"
    
    # Splits zum Verarbeiten
    splits = [
        ("training", extract_training_samples),
        ("testing", extract_test_samples)
    ]

    print("Starte EMNIST Digits Download und Konvertierung...")
    print(f"Zielverzeichnis: {output_dir}")

    for split_name, extractor in splits:
        print(f"\nVerarbeite {split_name}...")
        
        # Extrahiere Bilder und Labels (Die emnist-Library korrigiert die Spiegelung bereits automatisch!)
        images, labels = extractor('digits')
        
        # Erstelle Unterordner für jede Ziffer 0-9
        for i in range(10):
            (output_dir / split_name / str(i)).mkdir(parents=True, exist_ok=True)

        # Speichere Bilder
        for idx in tqdm(range(len(images)), desc=f"Speichere {split_name}"):
            img_array = images[idx]
            label = labels[idx]
            
            # Konvertiere numpy Array zu PIL Image
            img = Image.fromarray(img_array.astype('uint8'), mode='L')
            
            # Dateiname: emnist_{index}.png
            img.save(output_dir / split_name / str(label) / f"emnist_{idx}.png")

    print("\nFertig! Alle Bilder wurden in /data/emnist gespeichert.")
    print("EMNIST ist bereits 'MNIST-kompatibel' gedreht und gespiegelt worden.")

if __name__ == "__main__":
    try:
        save_emnist_to_folders()
    except Exception as e:
        print(f"\nFehler beim Download/Verarbeiten: {e}")
