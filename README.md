# KNN-Ziffernerkennung

## Projektstruktur

- `data/`
  - Trainingsdaten liegen hier.
  - Struktur: pro Ziffer ein Ordner `0` bis `9`.
  - In jedem Ordner sind durchnummerierte Bilddateien.
- `models/`
  - Hier werden trainierte Modelle gespeichert. (Naming sceme: TCM-oX mit anhänge -mini, normal also nichts und -pro)
- `train.py`
  - Lädt/liest die Trainingsdaten aus `data/`.
  - Trainiert das neuronale Netz.
  - Speichert das trainierte Modell nach `models/`.
  - Konfigurierbar (Iterationen und andere Modell einstellungen (mini pro normal und co.))
- `nn.py`
  - Enthält den gemeinsamen NN-Code (Netzwerk, Vorhersage, Laden/Speichern).
- `app.py`
  - Startet eine einfache Zeichen-UI.
  - Man kann ein Modell aus `models/` auswählen.
  - Zeichnung wird als Ziffer (0-9) vorhergesagt und angezeigt.
- `requirements.txt`
  - Benötigte Python-Pakete für Training und App.

## Ziel von `app.py`

`app.py` soll als einfache Benutzeroberfläche dienen: Zeichnen, Modell aus der directory wählen, Vorhersage anzeigen(zu wieviel prozent ist es welche Zahl).


