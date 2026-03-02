<!--
Dieses Dokument ist als "Briefing" gedacht, das du 1:1 in eine Präsentations-KI (Copilot/Gamma/Canva/Beautiful.ai/etc.)
kopieren kannst, damit sie daraus eine PowerPoint (PPTX) erzeugt.
-->

# PROMPT: Erstelle eine PowerPoint zu "TCM-AI Ziffernerkennung" (10 Minuten, 3 Personen)

Du bist eine Präsentations-KI und sollst aus den folgenden Projektdaten eine **fertige PowerPoint (PPTX)** erstellen.

## Output-Anforderungen

- Format: **PPTX**, 16:9, deutsch.
- Zielpublikum: Klasse/Dozent, gemischtes Tech-Level.
- Dauer: **10 Minuten total**.
- Sprecher: **3 Personen**:
  - Person A: nicht sehr technisch
  - Person B: nicht sehr technisch
  - Person C: technisch (kann Details zu NN/Adam/Datenpipeline/GPU erklären)
- Stil: klar, wenige Wörter pro Slide, visuell (Icons/Diagramme/Timeline), keine überladenen Tabellen auf Slides.
- Muss enthalten:
  - "Entwicklung über Zeit" als Story (Iterationsschritte)
  - Warum Adam Optimizer geholfen hat
  - Warum und wann auf größere Datensätze gewechselt wurde
  - Ergebnisse (Accuracy) und Vergleich der Modellfamilien
  - Kurzer Demo-Teil/Anwendungsfall (Zeichen-UI)

## Projekt-Kontext (kurz)

- Projektname: **TCM-AI-Ziffernerkennung**
- Aufgabe: Handschriftliche Ziffern (0-9) klassifizieren.
- Technische Vorgabe: neuronales Netz **selbst implementiert**, keine fertigen ML-Frameworks.
  - Nicht erlaubt: TensorFlow, PyTorch, Keras, scikit-learn
  - Erlaubt: NumPy, Matplotlib, Standardbibliotheken (GUI/Datei)
- Umsetzung:
  - Training/Inference CPU: `train.py`, `nn.py`
  - Optional schnelles GPU-Training: `train-gpu.py` (OpenCL via `pyopencl`)
  - Demo UI: `app.py` (Zeichenfläche, Modell wählen, Vorhersage + Wahrscheinlichkeiten)
- Modell-Artefakte pro Training in `models/`:
  - `.npz` Gewichte
  - `.json` Metadaten (Hyperparameter, Sample-Anzahl, finale Metriken)
  - `*_training.png` Trainingskurve (Loss/Accuracy)

## Datensätze (Daten-Story ist zentral)

1. Reduced MNIST (Kaggle) – genutzt in `TCM-o1` und `TCM-o2`
   - Grob: 10'000 Training / wenige Testing (in README als "Reduced-Dataset" beschrieben)
   - Link: https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist
2. MNIST PNG (Kaggle) – genutzt in `TCM-o3` und `TCM-o4`
   - 60'000 Training / 10'000 Testing
   - Link: https://www.kaggle.com/datasets/alexanderyyy/mnist-png
3. EMNIST Digits – genutzt ab `TCM-o5` (Download-Skript: `data/download_emnist.py`)
   - ca. 240'000 Training / 40'000 Test
   - Kombiniert mit MNIST Full zu ca. **300'000 Train / 50'000 Test**
   - Motivation: mehr Varianz an Handschriften, bessere Generalisierung

## Storyline: Entwicklung über die Zeit (Timeline)

Erzähle das Projekt als Iterationen, jeweils mit: Problem -> Änderung -> Effekt (Accuracy/Qualität/Speed).

### Iteration 1: `TCM-o1` (erste lauffähige Version)

- Datengröße: 8'000 Train / 2'000 Test (80/20 Split intern aus `data/`)
- Modell: 1 Hidden Layer, einfache Parameter, schnell trainierbar (Laptop)
- Ergebnis (finale Test-Accuracy):
  - `TCM-o1-mini`: 0.9185
  - `TCM-o1`: 0.9125
  - `TCM-o1-pro`: 0.9145
- Takeaway: Pipeline funktioniert, aber Generalisierung/Qualität begrenzt durch Datensatzgröße + Setup.

### Iteration 2: `TCM-o2` (Daten-Pipeline stabilisieren)

- Änderung ggü. `o1`:
  - Wechsel von internem Split auf feste Ordnerstruktur `data/training` + `data/testing`
  - Fester Seed (42) für reproduzierbare Ergebnisse
  - Parameter etwas optimiert
- Datengröße: 10'000 Train / 2'000 Test
- Ergebnis:
  - `TCM-o2-mini`: 0.9250
  - `TCM-o2`: 0.9370
  - `TCM-o2-pro`: 0.9485
- Takeaway: Reproduzierbarkeit + saubere Splits = verlässlicheres Tuning.

### Iteration 3: `TCM-o3` (Wechsel auf Full MNIST)

- Änderung ggü. `o2`:
  - Wechsel von Reduced (10k/2k) auf Full MNIST (60k/10k)
  - Parameter optimiert
- Ergebnis:
  - `TCM-o3-mini`: 0.9431
  - `TCM-o3`: 0.9526
  - `TCM-o3-pro`: 0.9603
- Takeaway: Größeres Dataset bringt sichtbar bessere Accuracy.

### Iteration 4: `TCM-o4` (mehr Hidden Layers)

- Änderung ggü. `o3`:
  - Statt 1 Hidden Layer nun 2-3 Hidden Layers (mehr Kapazität)
  - Optimiertere Parameter
- Ergebnis:
  - `TCM-o4-mini`: 0.9608
  - `TCM-o4`: 0.9652
  - `TCM-o4.1-pro`: 0.9832 (Sonderfall, siehe nächste Iteration)
- Takeaway: Mehr Kapazität + gutes Training bringt nochmal Schub.

### Iteration 4.1: `TCM-o4.1` (Adam Optimizer)

- Änderung:
  - Einführung **Adam** (Adaptive Moment Estimation) statt einfachem SGD.
- Aussage, die du in Slides erklären sollst:
  - Adam kombiniert Momentum + adaptive Lernraten pro Gewicht und konvergiert schneller/stabiler.
- Ergebnis sichtbar im Vergleich: `o4.1-pro` erreicht 0.9832 (Full MNIST, 3 Hidden Layers).

### Iteration 5: `TCM-o5` (Mega Dataset + Adam + Training-UX)

- Änderung ggü. `o4`:
  - Datensätze kombiniert: MNIST Full + EMNIST Digits => ca. 300'000 Train / 50'000 Test
  - Adam Optimizer als Standard
  - Live-Graphen während Training (Loss/Accuracy)
  - Early Stopping (patience 5)
- Ergebnis:
  - `TCM-o5-mini`: 0.9881
  - `TCM-o5`: 0.9902
  - `TCM-o5-pro`: 0.9901 (Notiz im Projekt: Benchmark wirkt schlechter als subjektive Qualität beim Ausprobieren)
- Takeaway: Datensatz-Varianz + Adam => starke Generalisierung.

### Iteration 5.1: `TCM-o5.1` (Data Augmentation)

- Änderung ggü. `o5`:
  - Data Augmentation eingebaut (z. B. Shift/Rotation)
- Ergebnis:
  - `TCM-o5.1-mini`: 0.9936
  - `TCM-o5.1`: 0.9938
  - `TCM-o5.1-pro`: 0.9937
- Takeaway: Augmentation erhöht Robustheit weiter, ohne neue Daten sammeln zu müssen.

## Technische Details (für Person C, aber nicht überladen)

### Modelltyp

- Einfache MLP / Fully-Connected NN (aus `nn.py`), Klassifikation 0-9.
- Training beinhaltet: Forward Pass, Loss, Backpropagation, Parameter-Updates.

### Adam in diesem Projekt (konkret)

- In `train.py` ist Adam Standard; Default Learning Rates:
  - mini/normal: 0.0015
  - pro: 0.0008
- Training hat optionale Features:
  - Early Stopping (CLI Default: 15; GPU-UI nutzt patience 5)
  - Data Augmentation Flags (Shift/Rotation, Probability)

### GPU-Training (OpenCL) als Engineering-Feature

- `train-gpu.py` nutzt `pyopencl` und eigene Kernel (Matrix-Multiplikationen usw.).
- Zweck: Größere Modelle/Datensätze werden CPU-seitig sehr langsam; GPU beschleunigt Training deutlich.
- Erzeugt kompatible Artefakte (`.npz`, `.json`, `*_training.png`) wie CPU-Training.

## Empfohlene Slide-Struktur (10 Minuten)

Erstelle 8 Slides + 1 Backup (optional). Gib pro Slide:
- Titel
- 3-6 Bulletpoints (kurz)
- Visual-Idee (Diagramm/Timeline/Icon)
- Sprecher (A/B/C)
- Sprecher-Notizen (30-60 Sekunden Text)

### Zeitplan / Rollen

- Person A (3:20):
  - Slide 1, 2, 3: Motivation, Ziel, Constraints, Überblick App
- Person B (3:20):
  - Slide 4, 5: Timeline Iterationen (o1->o3) und Dataset-Wechsel
- Person C (3:20):
  - Slide 6, 7, 8: Adam, Architektur/Layer, o4->o5.1, GPU-Training, Fazit & Ausblick

## Konkrete Slides (Inhalt, den du verwenden MUSST)

### Slide 1: Titel & Team

- Titel: "TCM-AI: Ziffernerkennung ohne ML-Frameworks"
- Untertitel: "Von 10k Samples zu 300k, von SGD zu Adam"
- Team: 3 Personen (Namen als Platzhalter: Person A/B/C)
- Visual: großes Ziffern-Icon + kleines NN-Icon

### Slide 2: Problem & Ziel

- Ziel: Ziffern 0-9 aus Bild klassifizieren
- Warum relevant: Handschrift variiert stark
- Output: Wahrscheinlichkeit pro Klasse, beste Vorhersage
- Visual: Beispiel-Ziffer + Output-Balkendiagramm (10 Klassen)

### Slide 3: Constraints & Setup (nicht zu technisch)

- "Netz selbst implementiert" (kein PyTorch/TensorFlow)
- CPU-Training (NumPy) + optional GPU (OpenCL)
- Demo-App: Zeichnen -> Modell wählen -> Vorhersage
- Visual: Architektur-Skizze (Data -> Train -> Model -> App)

### Slide 4: Timeline Teil 1 (o1 -> o2)

- o1: erster Prototyp, kleiner Datensatz, interne Splits
- o2: feste Train/Test-Ordner, reproduzierbar, bessere Accuracy
- Zahlen: o1-mini 0.9185, o2 0.9370, o2-pro 0.9485
- Visual: kleine Timeline + Accuracy-Pfeile nach oben

### Slide 5: Timeline Teil 2 (o3: Dataset-Sprung)

- Wechsel Reduced -> Full MNIST (60k/10k)
- Accuracy steigt auf 0.9526 / 0.9603 (o3/o3-pro)
- Kernaussage: "Datenmenge + Varianz = Generalisierung"
- Visual: Balken "Dataset size" + "Accuracy"

### Slide 6: o4 (mehr Layers) + Adam (o4.1)

- o4: 2-3 Hidden Layers, Accuracy 0.9652 (o4)
- o4.1-pro: Adam, 0.9832
- Warum Adam: schneller/stabiler, adaptive Lernraten, Momentum
- Visual: 2 Diagramme nebeneinander: "SGD vs Adam (schematisch)" + "Accuracy Jump"

### Slide 7: o5/o5.1 (Mega Dataset + Augmentation)

- Daten: MNIST Full + EMNIST Digits => ~300k/50k
- o5: ~0.990
- o5.1: ~0.994 (best: 0.9938)
- Augmentation: Shift/Rotation => robustere Inputs
- Visual: Beispiel Augmentation (shift/rotate) + Accuracy-Leaderboard

### Slide 8: Engineering & Fazit (inkl. GPU)

- GPU-Training via OpenCL für Geschwindigkeit (große Modelle/Datensätze)
- Artefakte: `.npz`, `.json`, `*_training.png`
- Fazit: Iteratives Vorgehen (Pipeline -> Daten -> Optimizer -> Augmentation)
- Ausblick: Confusion Matrix, bessere Normalisierung, mehr Augmentation, leichte CNN-ähnliche Features (nur falls erlaubt)
- Visual: "Lessons learned" 3 Kacheln + "Next steps" 3 Kacheln

### Backup Slide (optional): Modell-Übersicht (nur als Appendix)

Wenn du eine Tabelle machst, dann nur als Backup-Slide (nicht im Hauptteil zeigen).

## Wichtige Zahlen (zum Einbauen in Charts)

- Reduced-Dataset Phase:
  - o1: 8k/2k, best ca. 0.9185
  - o2: 10k/2k, best 0.9485
- Full MNIST Phase:
  - o3: 60k/10k, best 0.9603
  - o4: 60k/10k, 0.9652
  - o4.1-pro (Adam): 0.9832
- Mega Dataset + Adam + Augmentation:
  - o5: 300k/50k, best 0.9902
  - o5.1: 300k/50k, best 0.9938

## Letzte Anweisung

Erzeuge die PPTX so, dass sie ohne Nachbearbeitung vortragsfertig ist:
- klare Sprecher-Notizen je Slide
- konsistente Farben/Schrift
- Diagramme/Timelines statt Textwände
