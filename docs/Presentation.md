# Pr�sentation - TCM-AI Ziffernerkennung (10 Minuten, 3 Personen)

Ziel: Diese Vorlage ist so geschrieben, dass ihr sie einer KI geben k�nnt, damit sie daraus eine **PowerPoint** baut (Folien + Speaker Notes). Inhaltlich basiert sie auf `README.md` und `docs/TCM-AI_Dokumentation.docx`.

## Vorgaben f�r die PPT

- Sprache: Deutsch
- Dauer: **10 Minuten**
- Team: **3 Personen**
- Stil: klar, technisch korrekt, wenig Text, visuell
- Pro Folie: max. 5 Bulletpoints
- Speaker Notes: 2-4 S�tze
- Pflicht: 1 Folie mit Tabelle/Chart aus `docs/tcm_accuracy.xlsx`
- Kernaussagen:
  - selbst implementiertes NN (ohne ML-Frameworks)
  - Datenentwicklung von Start bis jetzt
  - "Gr��e ist nicht alles"

## Zeit- und Rollenplan (10:00)

- **Person 1 (0:00-3:20):** Folien 1-3
- **Person 2 (3:20-6:40):** Folien 4-6
- **Person 3 (6:40-10:00):** Folien 7-9

---

## Folie 1 - Titel & Ziel (Person 1)

**Titel:** TCM-AI: Handschriftliche Ziffernerkennung mit eigenem neuronalen Netz  
**Untertitel:** Von Reduced MNIST zu 300k+ Samples  
**Footer:** Tim, Mika, Cedric | EFIN26g | M�rz 2026

**Speaker Notes:**
Kurz vorstellen: Ziel war eine komplette Eigenimplementierung eines MLPs f�r Ziffern (0-9), ohne TensorFlow/PyTorch/Keras/scikit-learn.

## Folie 2 - Problem, Einschr�nkungen, Besonderheiten (Person 1)

- Input: 28x28 Graustufenbild (784 Features)
- Output: Klasse 0-9
- Kein fertiges ML-Framework erlaubt
- Alles selbst umgesetzt: Forward, Backprop, Training, Speicherung

**Speaker Notes:**
Hier die "speziellen Sachen": Wir haben nicht nur ein Modell trainiert, sondern den gesamten Lernprozess technisch selbst gebaut und nachvollziehbar gemacht.

## Folie 3 - Datens�tze: Start vs. Jetzt (Person 1)

- **Start (o1/o2):** Reduced MNIST, ca. 10k Train / 2k Test
- **Zwischenschritt (o3/o4):** MNIST Full, 60k Train / 10k Test
- **Jetzt (o5/o5.1):** MNIST + EMNIST kombiniert, ca. 300k Train / 50k Test
- Ergebnis: deutlich bessere Generalisierung

**Speaker Notes:**
Diese Folie ist zentral: Nicht nur Architektur, sondern vor allem Datenmenge und Datenvielfalt haben den gr��ten Qualit�tssprung gebracht.

## Folie 4 - Modellarchitektur & Profile (Person 2)

- MLP mit Fully Connected Layers
- Profile: mini / normal / pro
- 1 bis 3 Hidden Layers je nach Generation
- Parameterbereich: ~100k bis ~10M

**Speaker Notes:**
Erkl�rt, warum ihr Modellprofile nutzt: gleiche Idee, aber unterschiedliche Trade-offs bei Geschwindigkeit, Trainingsdauer und Accuracy.

## Folie 5 - Training & spezielle Technik (Person 2)

- Forward Pass -> Softmax
- Loss: Cross-Entropy
- Backpropagation + Mini-Batches
- Optimierer-Wechsel: von SGD zu Adam (ab o4.1/o5)

**Speaker Notes:**
Der Wechsel auf Adam war ein wichtiger technischer Schritt: stabilere und schnellere Konvergenz, besonders bei gr��eren Modellen und Datens�tzen.

## Folie 6 - Weitere Specials im Projekt (Person 2)

- GPU-Training mit `train-gpu.py` (OpenCL)
- Training-Monitoring (`*_training.png`)
- Early Stopping in sp�teren Generationen
- Data Augmentation ab o5.1

**Speaker Notes:**
Das sind eure "speziellen Sachen" neben der reinen Architektur: Engineering-Entscheidungen, die messbar bessere Resultate gebracht haben.

## Folie 7 - Ergebnisse aus Excel (Pflichtfolie) (Person 3)

**Inhalt aus `docs/tcm_accuracy.xlsx`:**

- Tabelle/Chart: Modellfamilien und Test-Accuracy
- Vergleich mini/normal/pro je Generation
- Optional zweite Achse: Parameterzahl

**Speaker Notes:**
Nur Testmetriken vergleichen und kurz sagen, dass alle Zahlen aus euren eigenen Trainingsl�ufen stammen.

## Folie 8 - Kernaussage: "Gr��e ist nicht alles" (Person 3)

- Gr��eres Modell != automatisch besser
- Wichtiger sind: Datenqualit�t, Datenvielfalt, Optimierer, Hyperparameter
- Beispiel aus euren Runs: pro oft nur knapp besser oder �hnlich
- o5.1 zeigt: Setup + Daten schl�gt reine Modellgr��e

**Speaker Notes:**
Bezieht euch direkt auf Folie 7 und nennt 1-2 konkrete Vergleichswerte aus Excel.

## Folie 9 - Fazit & Abschluss (Person 3)

- Endstand: bis ~99% Test-Accuracy
- Gr��te Fortschritte durch Datens�tze + Adam + sauberes Training
- Praktischer Beweis: App-Demo (`app.py`) bei Bedarf
- Takeaway: Verst�ndnis von NN von Grund auf aufgebaut

**Speaker Notes:**
Kurz, pr�zise Abschlussbotschaft. Wenn Zeit �brig bleibt: 30-60 Sek. Live-Demo oder Screenshot.

---

## Materialliste f�r die KI (PPT-Generator)

- `docs/tcm_accuracy.xlsx` (Ergebnisfolie)
- 1-2 Trainingsplots aus `models/*_training.png`
- optional Screenshot aus `app.py`
