# TCM-AI-Ziffernerkennung

## Projektstruktur

- `data/`
  - Enthält zwei Unterordner: `training/` und `testing/`.
  - Beide Unterordner haben jeweils die Unterordner `0` bis `9`.
  - In jedem Ziffernordner liegen die Bilddateien.
- `models/`
  - Hier werden trainierte Modelle gespeichert.
  - Naming-Schema: `TCM-o<version><size-tag>`
    - `<version>` ist eine Zahl (z. B. `1`, `2`, `3.1`).
    - `<size-tag>` ist optional:
      - kein Tag = `normal`
      - `-mini` = kleines/schnelles Modell
      - `-pro` = grösseres/langsames Modell
    - Beispiele:
      - `TCM-o1` (normal)
      - `TCM-o2-mini`
      - `TCM-o3-pro`
- `train.py`
  - Lädt/liest die Trainingsdaten aus `data/`.
  - Trainiert das neuronale Netz.
  - Speichert das trainierte Modell nach `models/`.
  - Konfigurierbar (Modellgrössen: mini/normal/pro).
- `train-gpu.py`
  - Eigenständiges GPU-Training per `pyopencl`.
  - Erstellt kompatible Artefakte (`.npz`, `.json`, `_training.png`) im gleichen Format wie `train.py`.
  - Bevorzugt automatisch GPU-Geräte und kann Plattform/Gerät explizit wählen.
- `nn.py`
  - Enthält den gemeinsamen NN-Code (Netzwerk, Vorhersage, Laden/Speichern).
- `app.py`
  - Startet eine einfache Zeichen-UI.
  - Modell kann aus `models/` ausgewählt werden.
  - Zeichnung wird als Ziffer (0-9) vorhergesagt und angezeigt.
- `requirements.txt`
  - Benötigte Python-Pakete für Training und App.

## Ziel von `app.py`

`app.py` soll als einfache Benutzeroberfläche dienen: Zeichnen, Modell aus dem Verzeichnis wählen, Vorhersage anzeigen (inkl. Prozentwahrscheinlichkeiten pro Zahl).

## Technische Vorgaben / Limitationen

- Das neuronale Netz muss selbst implementiert werden (kein fertiges ML-Framework).
- Nicht erlaubt: `tensorflow`, `pytorch`, `keras`, `scikit-learn`.
- Erlaubt: `numpy`, `matplotlib` sowie normale Hilfsbibliotheken für Datei/GUI.
- Trainingsprozess soll nachvollziehbar sein (Forward Pass, Fehlerberechnung, Backpropagation, Gradient Descent).
- Das Modell soll auf (reduced) MNIST Ziffernklassifikation ausgerichtet sein.

## Rechenbackend

- `app.py` und `train.py` laufen CPU-only mit `NumPy`.
- Für beschleunigtes Training auf einer GPU gibt es zusätzlich `train-gpu.py` mit OpenCL (`pyopencl`). (Ab o4-pro wäre die CPU in einem Desktopcomputer sehr langsam. per GPU geht es bedeutend schneller)

## Datenquelle

- Reduced MNIST (Kaggle): `https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist`
  - Verwendet in Modellfamilien `TCM-o1` und `TCM-o2`. (10'000 Training / few testing)
- MNIST PNG (Kaggle): `https://www.kaggle.com/datasets/alexanderyyy/mnist-png`
  - Verwendet in Modellfamilie `TCM-o3` und `TCM-o4` (60'000 Training / 10'000 Testing).
- EMNIST Digits (via `download_emnist.py`):
  - Verwendet ab Modellfamilie `TCM-o5`.
  - Enthält ca. 240'000 Training-Samples und 40'000 Test-Samples.
  - Bietet deutlich höhere Varianz in den Handschriften, was die Generalisierung verbessert.
  - Wird in zusammenspiel mit MNIST-Full verwendet für ein Trainings-Set mit 300'000 Samples.

## Adam-Optimizer

Ab Version 0.5 nutzen wir den **Adam-Optimizer** (Adaptive Moment Estimation). Im Vergleich zum Standard-SGD bietet er:
1. **Momentum**: Er merkt sich die Richtung der letzten Updates und überwindet so "lokale Minima" (Sackgassen) flüssiger.
2. **Adaptive Lernrate**: Er passt die Lernrate für jedes Gewicht individuell an.
3. **Konvergenz**: Das Modell erreicht viel schneller (in deutlich weniger Epochen) eine hohe Genauigkeit.

###

## Modelle

Alle Modelle liegen in `models/` als:

- `.npz` – Gewichte
- `.json` – Metadaten (u. a. Samples + Hyperparameter + finale Metriken)
- `*_training.png` – Trainingskurve (Loss/Accuracy)

### Familie `TCM-o1` (First test trained in seconds on a Laptop)

- Daten: 10'000 total → 8'000 Train / 2'000 Test (80/20 Split aus `data/`).

| Modell        | Daten (Train/Test) | hidden | epochs | batch size | Learning Rate | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `TCM-o1-mini` | 8'000 / 2'000      | 128    | 128    | 256   | 0.008 | 0.9185          |
| `TCM-o1`      | 8'000 / 2'000      | 512    | 256    | 256   | 0.005 | 0.9125          |
| `TCM-o1-pro`  | 8'000 / 2'000      | 2048   | 512    | 512   | 0.003 | 0.9145          |

### Familie `TCM-o2` (Optimierte o1, könnte auf o1.1 genannt worden sein...)

- Daten: 10'000 Train / 2'000 Test (aus `data/training` und `data/testing`).
- Änderung ggü. `TCM-o1`: Wechsel von internem Split auf feste Ordner und fester Seed (42). Optimiertere Parameter.

| Modell        | Daten (Train/Test) | hidden | epochs | batch size | Learning Rate | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `TCM-o2-mini` | 10'000 / 2'000     | 128    | 64     | 256   | 0.008 | 0.9250          |
| `TCM-o2`      | 10'000 / 2'000     | 512    | 128    | 256   | 0.005 | 0.9370          |
| `TCM-o2-pro`  | 10'000 / 2'000     | 2048   | 512    | 512   | 0.003 | 0.9485          |

### Familie `TCM-o3` (FULL DATASET and better Parameters)

- Daten: 60'000 Train / 10'000 Test (aus `data/training` und `data/testing`).
- Änderung ggü. `TCM-o2`: Wechsel von Reduced-Dataset (10'000/2'000) auf Full-Dataset (60'000/10'000). Optimiertere Parameter.

| Modell        | Daten (Train/Test) | hidden | epochs | batch size | Learning Rate    | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `TCM-o3-mini` | 60'000 / 10'000    | 128    | 64     | 256   | 0.008 | 0.9431          |
| `TCM-o3`      | 60'000 / 10'000    | 512    | 128    | 256   | 0.005 | 0.9526          |
| `TCM-o3-pro`  | 60'000 / 10'000    | 2048   | 512    | 512   | 0.003 | 0.9603          |

### Familie `TCM-o4` (Full Dataset, hidden layers)

- Daten: 60'000 Train / 10'000 Test (aus `data/training` und `data/testing`).
- Änderung ggü. `TCM-o3`:  Optimiertere Parameter. Anstatt 1 nun 2-3 Hidden Layers.

| Modell        | Daten (Train/Test) | hidden | epochs | batch size | Learning Rate | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `TCM-o4-mini` | 60'000 / 10'000    | 256    | 96     | 128   | 0.0025 | 0.9608 |
| `TCM-o4`      | 60'000 / 10'000    | 512    | 192    | 128   | 0.0015 | 0.9652          |

### Familie `TCM-o5` (Mega Dataset & Adam Optimizer)

- Daten: Kombiniertes Set aus MNIST Full + EMNIST Digits (~300'000 Train / 50'000 Test).
- Änderung ggü. `TCM-o4`: 
  - **Adam Optimizer**: Wechsel vom einfachen SGD auf den Adam-Optimizer.
  - **Live-Graphen während dem trainieren**: Echtzeit-Visualisierung von Loss und Accuracy während dem trainieren.
