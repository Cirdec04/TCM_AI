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
      - `-pro` = gröesseres/langsames Modell
    - Beispiele:
      - `TCM-o1` (normal)
      - `TCM-o2-mini`
      - `TCM-o3-pro`
- `train.py`
  - Lädt/liest die Trainingsdaten aus `data/`.
  - Trainiert das neuronale Netz.
  - Speichert das trainierte Modell nach `models/`.
  - Konfigurierbar (Modellgrössen wie mini/normal/pro).
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

## Datenquelle

- Reduced MNIST (Kaggle): `https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist`
  - Verwendet in Modellfamilien `TCM-o1` und `TCM-o2`. (10'000 Training / few testing)
- MNIST PNG (Kaggle): `https://www.kaggle.com/datasets/alexanderyyy/mnist-png`
  - Verwendet in Modellfamilie `TCM-o3` und later (60'000 Training / 10'000 Testing).

## Modelle

Alle Modelle liegen in `models/` als:

- `.npz` – Gewichte
- `.json` – Metadaten (u. a. Samples + Hyperparameter + finale Metriken)
- `*_training.png` – Trainingskurve (Loss/Accuracy)

### Familie `TCM-o1` (First test trained in seconds on a Laptop)

- Daten: 10'000 total → 8'000 Train / 2'000 Test (80/20 Split aus `data/`).

| Modell        | Daten (Train/Test) | hidden | epochs | batch | lr    | finale Test-Acc |
|---|---:|---:|---:|---:|---:|---:|
| `TCM-o1-mini` | 8'000 / 2'000      | 128    | 128    | 256   | 0.008 | 0.9185          |
| `TCM-o1`      | 8'000 / 2'000      | 512    | 256    | 256   | 0.005 | 0.9125          |
| `TCM-o1-pro`  | 8'000 / 2'000      | 2048   | 512    | 512   | 0.003 | 0.9145          |

### Familie `TCM-o2` (Optimierte o1, könnte auf o1.1 genannt worden sein...)

- Daten: 10'000 Train / 2'000 Test (aus `data/training` und `data/testing`).
- Änderung ggü. `TCM-o1`: Wechsel von internem Split auf feste Ordner und fester Seed (42). Optimiertere Parameter.

| Modell        | Daten (Train/Test) | hidden | epochs | batch | lr    | finale Test-Acc |
|---|---:|---:|---:|---:|---:|---:|
| `TCM-o2-mini` | 10'000 / 2'000     | 128    | 64     | 256   | 0.008 | 0.9250          |
| `TCM-o2`      | 10'000 / 2'000     | 512    | 128    | 256   | 0.005 | 0.9370          |
| `TCM-o2-pro`  | 10'000 / 2'000     | 2048   | 512    | 512   | 0.003 | 0.9485          |

### Familie `TCM-o3` (FULL DATASET and better Parameters)

- Daten: 60'000 Train / 10'000 Test (aus `data/training` und `data/testing`).
- Änderung ggü. `TCM-o2`: Wechsel von Reduced-Dataset (10'000/2'000) auf Full-Dataset (60'000/10'000). Optimiertere Parameter.

| Modell        | Daten (Train/Test) | hidden | epochs | batch | lr    | finale Test-Acc |
|---|---:|---:|---:|---:|---:|---:|
| `TCM-o3-mini` | 60'000 / 10'000    | 128    | 64     | 256   | 0.008 | 0.9431          |
| `TCM-o3`      | 60'000 / 10'000    | 512    | 128    | 256   | 0.005 | 0.9526          |
| `TCM-o3-pro`  | 60'000 / 10'000    | 2048   | 512    | 512   | 0.003 | 0.9603          |
