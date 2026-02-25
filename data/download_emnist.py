import os
import requests
import zipfile
import gzip
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def download_file_from_google_drive(id, destination):
    print(f"[1/5] Verbinde mit Google Drive...")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        print("[1/5] Virenwarnung von Google erkannt. Bestaetige Download...")
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    print(f"[2/5] Starte Download (Groesse: {total_size / (1024*1024):.2f} MB)...")
    
    with open(destination, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc="EMNIST Download"
    ) as progress:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))
    print(f"\n[2/5] Download abgeschlossen.")

def parse_idx(fd):
    header = fd.read(4)
    if len(header) < 4: return None
    magic, data_type, dims = header[0:2], header[2], header[3]
    shape = tuple(int.from_bytes(fd.read(4), 'big') for _ in range(dims))
    return np.frombuffer(fd.read(), dtype=np.uint8).reshape(shape)

def save_emnist_to_folders():
    base_dir = Path(__file__).resolve().parent
    temp_dir = base_dir / "temp_emnist"
    output_dir = base_dir / "data" / "EMNIST"  # Grossbuchstaben wie der vorhandene Ordner
    zip_path = base_dir / "emnist.zip"

    temp_dir.mkdir(exist_ok=True)

    # 1. Download (beschädigte ZIP vorab prüfen und ggf. löschen)
    if zip_path.exists():
        if zip_path.stat().st_size < 100_000_000:  # kleiner als 100 MB = kaputt
            print(f"[1/5] Beschädigte emnist.zip gefunden ({zip_path.stat().st_size // 1024} KB). Lösche und lade neu herunter...")
            zip_path.unlink()
        else:
            print(f"[1/5] emnist.zip gefunden ({zip_path.stat().st_size // (1024*1024)} MB). Überspringe Download.")

    if not zip_path.exists():
        download_file_from_google_drive('1R0blrtCsGEVLjVL3eijHMxrwahRUDK26', str(zip_path))
    
    # 2. Entpacken
    print(f"[3/5] Entpacke emnist.zip nach {temp_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        print(f"      Gefundene Dateien im Archiv: {len(file_list)}")
        zf.extractall(temp_dir)
    print("      Entpacken fertig.")

    # 3. Verarbeiten
    data_map = {
        "training": ("emnist-digits-train-images-idx3-ubyte.gz", "emnist-digits-train-labels-idx1-ubyte.gz"),
        "testing": ("emnist-digits-test-images-idx3-ubyte.gz", "emnist-digits-test-labels-idx1-ubyte.gz")
    }

    print(f"[4/5] Verarbeite Binärdaten und konvertiere zu PNG...")
    for split, (img_file, lbl_file) in data_map.items():
        img_path = temp_dir / "gzip" / img_file
        lbl_path = temp_dir / "gzip" / lbl_file

        print(f"      Lese {img_file}...")
        with gzip.open(img_path, 'rb') as f:
            images = parse_idx(f)
            images = images.swapaxes(1, 2)
        
        print(f"      Lese {lbl_file}...")
        with gzip.open(img_path.parent / lbl_file, 'rb') as f:
            labels = parse_idx(f)

        print(f"      Speichere {split} Samples ({len(images)} Bilder)...")
        for i in tqdm(range(len(images))):
            label = labels[i]
            target_path = output_dir / split / str(label)
            target_path.mkdir(parents=True, exist_ok=True)
            
            img = Image.fromarray(images[i], mode='L')
            img.save(target_path / f"emnist_sample_{i}.png")

    # Aufräumen
    print("\nAufräumen...")
    import shutil
    shutil.rmtree(temp_dir)
    if zip_path.exists():
        os.remove(zip_path)

    print("\nFertig! Alle EMNIST Digits wurden in /data/emnist gespeichert.")

if __name__ == "__main__":
    save_emnist_to_folders()
