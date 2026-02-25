# Präsentation (KI-Vorlage) – TCM-AI Ziffernerkennung (5–10 Minuten)

Ziel: Diese Vorlage ist so geschrieben, dass ihr sie einer KI geben könnt, damit sie daraus eine **PowerPoint** baut (mit Folien + Speaker Notes). Wichtige Daten/Charts kommen aus `docs/tcm_accuracy.xlsx`.

## Vorgaben für die KI (PPT-Erstellung)

- Sprache: Deutsch
- Dauer: 5–10 Minuten
- Stil: klar, technisch korrekt, nicht zu textlastig
- Pro Folie: max. 5 Bulletpoints
- Dazu Speaker Notes: 2–4 Sätze pro Folie
- Einbauen: 1 Folie mit Excel-Tabelle/Chart aus `docs/tcm_accuracy.xlsx`
- Kernaussage: „**Größe ist nicht alles**“ (mehr Parameter ≠ automatisch bessere Accuracy)

## Folie 1 — Titel

**Titel:** Ziffernerkennung mit einem selbst implementierten neuronalen Netz (MLP)  
**Untertitel:** TCM-AI – MNIST / Reduced MNIST  
**Footer:** Team, Klasse, Datum

**Speaker Notes:**  
Kurz vorstellen: Wir haben ein neuronales Netz ohne ML-Frameworks gebaut und trainiert, das handgeschriebene Ziffern erkennt.

## Folie 2 — Problem & Ziel

- Input: 28×28 Graustufenbild
- Output: Ziffer 0–9
- Ziel: hohe Test-Accuracy
- Einschränkung: kein TensorFlow/PyTorch/Keras/scikit-learn

**Speaker Notes:**  
Erklären, was Klassifikation ist und warum ein train/test Split wichtig ist.

## Folie 3 — Daten & Vorverarbeitung

- Datenstruktur: `data/training/0..9`, `data/testing/0..9`
- Graustufen, Resize 28×28
- Normalisierung auf `[0,1]`
- Flatten → 784 Features

**Speaker Notes:**  
Betonen, dass saubere Vorverarbeitung und eine feste Struktur die Reproduzierbarkeit erhöht.

## Folie 4 — Künstliches Neuron (Basics)

- Formel: `z = w·x + b`
- Aktivierung: `a = f(z)`
- Hidden: ReLU
- Output: Softmax → Wahrscheinlichkeiten

**Speaker Notes:**  
Kurz: Ohne Aktivierungsfunktion wäre es nur linear; Softmax macht aus Scores echte Wahrscheinlichkeiten.

## Folie 5 — Unser Modell (MLP)

- Fully-Connected Layers
- 784 → Hidden → 10
- Profile: mini / normal / pro (Speed vs. Accuracy)
- Mehr Layers/Neuronen = mehr Parameter

**Speaker Notes:**  
Einfach erklären: MLP kann komplexere Muster lernen als ein einzelnes Neuron.

## Folie 6 — Training (Was passiert pro Batch?)

- Forward Pass → `p`
- Loss: Cross-Entropy
- Backpropagation → Gradienten
- Update: Adam Optimizer

**Speaker Notes:**  
Trainingsprozess „nachvollziehbar“ erklären: Wir berechnen Fehler, leiten ihn rückwärts ab und passen Gewichte an.

## Folie 7 — Visualisierung / Monitoring

- Train/Test Loss
- Train/Test Accuracy
- Plots werden gespeichert (`models/*_training.png`)
- Nutzt Plots zum Erkennen von Over-/Underfitting

**Speaker Notes:**  
Beispiel erklären: Wenn Train-Accuracy steigt, Test aber fällt → Overfitting-Verdacht.

## Folie 8 — Ergebnisse aus Excel (Pflichtfolie)

**Inhalt dieser Folie (aus `docs/tcm_accuracy.xlsx`):**

- Tabelle oder Chart: Modelle vs. Test-Accuracy
- Zusätzlich: Parameteranzahl oder „Modellgröße“ (mini/normal/pro)

**Speaker Notes:**  
Sagt, woher die Zahlen kommen (eigene Runs) und dass ihr bewusst Testwerte vergleicht.

## Folie 9 — „Größe ist nicht alles“ (Kernaussage)

- Größere Netze ≠ automatisch bessere Test-Accuracy
- Gründe (je nach euren Daten):
  - Overfitting
  - Hyperparameter nicht optimal
  - zu wenig/zu wenig diverse Daten
  - Trainingstabilität/Plateaus

**Speaker Notes:**  
Bezieht euch direkt auf Folie 8: zeigt ein Beispiel, wo ein kleineres Modell ähnlich gut oder besser ist. Fazit: Architektur + Daten + Training müssen zusammenpassen.

## Folie 10 — Demo (optional, wenn Zeit)

- `python app.py`
- Ziffer zeichnen
- Vorhersage + Wahrscheinlichkeiten anzeigen

**Speaker Notes:**  
Kurz live zeigen, dass das Modell im Alltag „greifbar“ ist. Wenn keine Zeit: Screenshot/kurzer Ablauf.

## Folie 11 — Learnings & Fazit

- Wichtigste Learnings (3–4 Punkte)
- Was wir verbessert haben (z. B. Adam, mehr Daten, mehr Layers)
- Nächste Schritte (z. B. bessere Augmentation/Regularisierung)

**Speaker Notes:**  
Mit einem Satz abschließen: Wir verstehen jetzt, wie Neuronen, Training und Evaluation zusammenspielen.

## Materialliste für die KI (damit sie die PPT baut)

- `docs/tcm_accuracy.xlsx` (Zahlen + ggf. Chart übernehmen)
- 1–2 Trainingsplots aus `models/*_training.png` (als Bilder einfügen)
- Optional: Screenshot der App (`app.py`) für die Demo-Folie

