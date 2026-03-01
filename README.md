# TCM-AI-Ziffernerkennung

## Abgabe-Artefakte

- Dokumentation (Word): `docs/TCMAI-Dokumentation.docx`
- Präsentation (PowerPoint): `docs/TCMAI-Präsentation.pptx`
- Auswertung (Excel): `docs/tcm_accuracy.xlsx`

## Projektstruktur

- `data/`
  - Enth�lt zwei Unterordner: `training/` und `testing/`.
  - Beide Unterordner haben jeweils die Unterordner `0` bis `9`.
  - In jedem Ziffernordner liegen die Bilddateien.
- `models/`
  - Hier werden trainierte Modelle gespeichert.
  - Naming-Schema: `TCM-o<version><size-tag>`
    - `<version>` ist eine Zahl (z. B. `1`, `2`, `3.1`).
    - `<size-tag>` ist optional:
      - kein Tag = `normal`
      - `-mini` = kleines/schnelles Modell
      - `-pro` = gr�sseres/langsames Modell
    - Beispiele:
      - `TCM-o1` (normal)
      - `TCM-o2-mini`
      - `TCM-o3-pro`
- `train.py`
  - L�dt/liest die Trainingsdaten aus `data/`.
  - Trainiert das neuronale Netz.
  - Speichert das trainierte Modell nach `models/`.
  - Konfigurierbar (Modellgr�ssen: mini/normal/pro).
- `train-gpu.py`
  - Eigenst�ndiges GPU-Training per `pyopencl`.
  - Erstellt kompatible Artefakte (`.npz`, `.json`, `_training.png`) im gleichen Format wie `train.py`.
  - Bevorzugt automatisch GPU-Ger�te und kann Plattform/Ger�t explizit w�hlen.
- `nn.py`
  - Enth�lt den gemeinsamen NN-Code (Netzwerk, Vorhersage, Laden/Speichern).
- `app.py`
  - Startet eine einfache Zeichen-UI.
  - Modell kann aus `models/` ausgew�hlt werden.
  - Zeichnung wird als Ziffer (0-9) vorhergesagt und angezeigt.
- `requirements.txt`
  - Ben�tigte Python-Pakete f�r Training und App.

## Ziel von `app.py`

`app.py` soll als einfache Benutzeroberfl�che dienen: Zeichnen, Modell aus dem Verzeichnis w�hlen, Vorhersage anzeigen (inkl. Prozentwahrscheinlichkeiten pro Zahl).

## Technische Vorgaben / Limitationen

- Das neuronale Netz muss selbst implementiert werden (kein fertiges ML-Framework).
- Nicht erlaubt: `tensorflow`, `pytorch`, `keras`, `scikit-learn`.
- Erlaubt: `numpy`, `matplotlib` sowie normale Hilfsbibliotheken f�r Datei/GUI.
- Trainingsprozess soll nachvollziehbar sein (Forward Pass, Fehlerberechnung, Backpropagation, Gradient Descent).
- Das Modell soll auf (reduced) MNIST Ziffernklassifikation ausgerichtet sein.

## Rechenbackend

- `app.py` und `train.py` laufen CPU-only mit `NumPy`.
- F�r beschleunigtes Training auf einer GPU gibt es zus�tzlich `train-gpu.py` mit OpenCL (`pyopencl`). (Ab o4-pro w�re die CPU in einem Desktopcomputer sehr langsam. per GPU geht es bedeutend schneller)

## Datenquelle

- Reduced MNIST (Kaggle): `https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist`
  - Verwendet in Modellfamilien `TCM-o1` und `TCM-o2`. (10'000 Training / few testing)
- MNIST PNG (Kaggle): `https://www.kaggle.com/datasets/alexanderyyy/mnist-png`
  - Verwendet in Modellfamilie `TCM-o3` und `TCM-o4` (60'000 Training / 10'000 Testing).
- EMNIST Digits (via `data/download_emnist.py`):
  - Verwendet ab Modellfamilie `TCM-o5`.
  - Enth�lt ca. 240'000 Training-Samples und 40'000 Test-Samples.
  - Bietet deutlich h�here Varianz in den Handschriften, was die Generalisierung verbessert.
  - Wird in zusammenspiel mit MNIST-Full verwendet f�r ein Trainings-Set mit 300'000 Samples.

## Adam-Optimizer

Ab Version 0.5 nutzen wir den **Adam-Optimizer** (Adaptive Moment Estimation). Im Vergleich zum Standard-SGD bietet er:
1. **Momentum**: Er merkt sich die Richtung der letzten Updates und �berwindet so "lokale Minima" (Sackgassen) fl�ssiger.
2. **Adaptive Lernrate**: Er passt die Lernrate f�r jedes Gewicht individuell an.
3. **Konvergenz**: Das Modell erreicht viel schneller (in deutlich weniger Epochen) eine hohe Genauigkeit.

###

## Modelle

Alle Modelle liegen in `models/` als:

- `.npz` � Gewichte
- `.json` � Metadaten (u. a. Samples + Hyperparameter + finale Metriken)
- `*_training.png` � Trainingskurve (Loss/Accuracy)

### Familie `TCM-o1` (First test trained in seconds on a Laptop)

- Daten: 10'000 total ? 8'000 Train / 2'000 Test (80/20 Split aus `data/`).

| Modell        | Daten (Train/Test) | hidden | hidden layers | parameters | epochs | batch size | Learning Rate | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `TCM-o1-mini` | 8'000 / 2'000      | 128    | 1 | 101.77K | 128    | 256   | 0.008 | 0.9185          |
| `TCM-o1`      | 8'000 / 2'000      | 512    | 1 | 407.05K | 256    | 256   | 0.005 | 0.9125          |
| `TCM-o1-pro`  | 8'000 / 2'000      | 2048   | 1 | 1.63M | 512    | 512   | 0.003 | 0.9145          |

### Familie `TCM-o2` (Optimierte o1, k�nnte auf o1.1 genannt worden sein...)

- Daten: 10'000 Train / 2'000 Test (aus `data/training` und `data/testing`).
- �nderung gg�. `TCM-o1`: Wechsel von internem Split auf feste Ordner und fester Seed (42). Optimiertere Parameter.

| Modell        | Daten (Train/Test) | hidden | hidden layers | parameters | epochs | batch size | Learning Rate | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `TCM-o2-mini` | 10'000 / 2'000     | 128    | 1 | 101.77K | 64     | 256   | 0.008 | 0.9250          |
| `TCM-o2`      | 10'000 / 2'000     | 512    | 1 | 407.05K | 128    | 256   | 0.005 | 0.9370          |
| `TCM-o2-pro`  | 10'000 / 2'000     | 2048   | 1 | 1.63M | 512    | 512   | 0.003 | 0.9485          |

### Familie `TCM-o3` (FULL DATASET and better Parameters)

- Daten: 60'000 Train / 10'000 Test (aus `data/training` und `data/testing`).
- �nderung gg�. `TCM-o2`: Wechsel von Reduced-Dataset (10'000/2'000) auf Full-Dataset (60'000/10'000). Optimiertere Parameter.

| Modell        | Daten (Train/Test) | hidden | hidden layers | parameters | epochs | batch size | Learning Rate    | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `TCM-o3-mini` | 60'000 / 10'000    | 128    | 1 | 101.77K | 64     | 256   | 0.008 | 0.9431          |
| `TCM-o3`      | 60'000 / 10'000    | 512    | 1 | 407.05K | 128    | 256   | 0.005 | 0.9526          |
| `TCM-o3-pro`  | 60'000 / 10'000    | 2048   | 1 | 1.63M | 512    | 512   | 0.003 | 0.9603          |

### Familie `TCM-o4` (Full Dataset, hidden layers)

- Daten: 60'000 Train / 10'000 Test (aus `data/training` und `data/testing`).
- �nderung gg�. `TCM-o3`:  Optimiertere Parameter. Anstatt 1 nun 2-3 Hidden Layers.

| Modell        | Daten (Train/Test) | hidden | hidden layers | parameters | epochs | batch size | Learning Rate | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `TCM-o4-mini` | 60'000 / 10'000    | 256    | 2 | 269.32K | 96     | 128   | 0.0025 | 0.9608 |
| `TCM-o4`      | 60'000 / 10'000    | 512    | 2 | 669.71K | 192    | 128   | 0.0015 | 0.9652          |
| `TCM-o4.1-pro`| 60'000 / 10'000    | 2048   | 3 | 10.02M | 42    | 512   | Adam   | 0.9832          |

Notiz zu `TCM-o4.1`:
- Nutzt bereits den **Adam-Optimizer**.
- Alle anderen Kern-Einstellungen (Datenbasis, Modellprofil/Familie `o4`) bleiben wie bei `o4`.

### Familie `TCM-o5` (Mega Dataset & Adam Optimizer)

- Daten: Kombiniertes Set aus MNIST Full + EMNIST Digits (~300'000 Train / 50'000 Test).
- Änderung ggü. `TCM-o4`: 
  - **Adam Optimizer**: Wechsel vom einfachen SGD auf den Adam-Optimizer.
  - **Live-Graphen w�hrend dem trainieren**: Echtzeit-Visualisierung von Loss und Accuracy w�hrend dem trainieren.
  - Early Stopping mit patience 5

| Modell        | Daten (Train/Test) | hidden | hidden layers | parameters | epochs | batch size | Learning Rate | finale Test-Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `TCM-o5-mini` | 300'000 / 50'000   | 256    | 2 | 269.3K | 11     | 512   | Adam | 0.9881 |
| `TCM-o5`      | 300'000 / 50'000   | 512    | 2 | 669.7K | 19     | 512   | Adam | 0.9902 |
| `TCM-o5-pro`  | 300'000 / 50'000   | 2048   | 3 | 10.02M | 18     | 512   | Adam | 0.9901 |
| `TCM-o5.1-mini` | 300'000 / 50'000 | 256    | 2 | 269.3K | 44     | 512   | Adam | 0.9936 |
| `TCM-o5.1`      | 300'000 / 50'000 | 512    | 2 | 669.7K | 40     | 512   | Adam | 0.9938 |
| `TCM-o5.1-pro`  | 300'000 / 50'000 | 2048   | 3 | 10.02M | 22     | 512   | Adam | 0.9937 |

Notiz zu `TCM-o5-pro`:
Benchmark zeigt es schlechter an als es sich anfühlt. Beim ausprobieren war es merkbar besser als o5.

Notiz zu `TCM-o5-Familie`:
Alle Modelle werden manuell noch getestet und die o5-Familie erkennt extrem viel mehr als alle vorderen. dies liegt vermutlich vor allem am grösseren Trainingsdatensatz.

Notiz zu `TCM-o5.1-Familie`:
Data Augmentation eingebaut.
