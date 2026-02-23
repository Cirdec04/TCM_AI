# KNN-Ziffernerkennung

## Projektstruktur

- `data/`
  - Enthält zwei Unterordner: `training/` und `testing/`.
  - Beide Unterordner haben jeweils die Unterordner `0` bis `9`.
  - In jedem Ziffernordner liegen die Bilddateien.
- `models/`
  - Hier werden trainierte Modelle gespeichert.
  - Naming-Schema: `TCM-o<version><size-tag>`
    - `<version>` ist eine ganze Zahl (z. B. `1`, `2`, `3`).
    - `<size-tag>` ist optional:
      - kein Tag = `normal`
      - `-mini` = kleines/schnelles Modell
      - `-pro` = gräesseres/langsames Modell
      - `-max` = sehr grosses/sehr langsames Modell
    - Beispiele:
      - `TCM-o1` (normal)
      - `TCM-o2-mini`
      - `TCM-o3-pro`
      - `TCM-o4-max`
- `train.py`
  - Laedt/liest die Trainingsdaten aus `data/`.
  - Trainiert das neuronale Netz.
  - Speichert das trainierte Modell nach `models/`.
  - Konfigurierbar (Iterationen, Modellgroessen wie mini/normal/pro/max, etc.).
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

- Reduced MNIST: `https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist` / gespeichert unter `data/training` und `data/testing`
