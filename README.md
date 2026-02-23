# KNN-Ziffernerkennung (einfaches Python-Projekt)

## Projektstruktur

- `data/`
  - Trainingsdaten liegen hier.
  - Struktur: pro Ziffer ein Ordner `0` bis `9`.
  - In jedem Ordner sind durchnummerierte Bilddateien (z. B. `1.png`, `2.png`, ...).
- `models/`
  - Hier werden trainierte Modelle gespeichert (z. B. `.npz` Dateien).
- `train.py`
  - L�dt/liest die Trainingsdaten aus `data/`.
  - Trainiert das neuronale Netz.
  - Speichert das trainierte Modell nach `models/`.
- `nn.py`
  - Enth�lt den gemeinsamen NN-Code (Netzwerk, Vorhersage, Laden/Speichern).
- `app.py`
  - Startet eine einfache Zeichen-UI.
  - Man kann ein Modell aus `models/` ausw�hlen.
  - Zeichnung wird als Ziffer (0-9) vorhergesagt und angezeigt.
- `requirements.txt`
  - Ben�tigte Python-Pakete f�r Training und App.

## Ziel von `app.py`

`app.py` soll als einfache Benutzeroberfl�che dienen: Zeichnen, Modell aus der directory w�hlen, Vorhersage anzeigen.
