from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from deps import ensure_requirements_installed

ensure_requirements_installed()

import numpy as np

from nn import SimpleMLP

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


class DigitApp:
    def __init__(self, root: tk.Tk, models_dir: Path) -> None:
        self.root = root
        self.root.title("TCM AI Digit Recognizer")
        self.root.resizable(False, False)

        self.models_dir = models_dir
        self.grid_size = 28
        self.display_scale = 12
        self.canvas_size = self.grid_size * self.display_scale
        self.brush_radius = 2
        self.brush_strength = 0.35

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.pixel_ids: list[list[int]] = []
        self.model: SimpleMLP | None = None
        self.model_name: str | None = None
        self.loaded_models: dict[str, SimpleMLP] = {}

        self.model_var = tk.StringVar()
        self.test_all_var = tk.BooleanVar(value=False)
        self.result_var = tk.StringVar(value="Noch keine Idee.")
        self.all_models_var = tk.StringVar(value="Test all ist aus.")

        self._build_ui()
        self.refresh_model_list()

    def _build_ui(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Modell:").pack(side="left")
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, state="readonly", width=30)
        self.model_combo.pack(side="left", padx=8)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)

        ttk.Button(top_frame, text="Refresh", command=self.refresh_model_list).pack(side="left")
        ttk.Checkbutton(
            top_frame,
            text="Test all",
            variable=self.test_all_var,
            command=self.on_toggle_test_all,
        ).pack(side="left", padx=(10, 0))

        content_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        content_frame.pack()

        canvas_frame = ttk.Frame(content_frame)
        canvas_frame.grid(row=0, column=0, sticky="n")

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=1)
        self.canvas.pack()
        self._init_pixel_canvas()
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<Button-1>", self.on_draw)
        self.canvas.bind("<B3-Motion>", self.on_erase)
        self.canvas.bind("<Button-3>", self.on_erase)
        ttk.Label(
            canvas_frame,
            text=f"Zoom x{self.display_scale} (28x28 Raster) | Linksklick: weicher Brush | Rechtsklick: radieren",
        ).pack(pady=(6, 0))

        side_frame = ttk.LabelFrame(content_frame, text="Alle Modelle", padding=8)
        side_frame.grid(row=0, column=1, sticky="nw", padx=(12, 0))
        ttk.Label(side_frame, textvariable=self.all_models_var, justify="left", width=36).pack(anchor="nw")

        buttons = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")

        ttk.Button(buttons, text="Alles loeschen", command=self.clear_canvas).pack(side="left")

        result_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        result_frame.pack(fill="x")
        ttk.Label(result_frame, textvariable=self.result_var, justify="left").pack(anchor="w")

    def _init_pixel_canvas(self) -> None:
        self.pixel_ids = []
        for y in range(self.grid_size):
            row_ids: list[int] = []
            for x in range(self.grid_size):
                x1 = x * self.display_scale
                y1 = y * self.display_scale
                x2 = x1 + self.display_scale
                y2 = y1 + self.display_scale
                pixel_id = self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill="black",
                    outline="#1a1a1a",
                    width=1,
                    tags=("pixel",),
                )
                row_ids.append(pixel_id)
            self.pixel_ids.append(row_ids)

    def refresh_model_list(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_files = sorted([p.name for p in self.models_dir.glob("TCM-o*.npz")])
        self.model_combo["values"] = model_files

        if not model_files:
            self.model_var.set("")
            self.model = None
            self.model_name = None
            self.loaded_models.clear()
            self.result_var.set("Keine Modelle gefunden")
            self.all_models_var.set("Keine Modelle gefunden.")
            return

        self.loaded_models = {name: model for name, model in self.loaded_models.items() if name in model_files}

        if self.model_var.get() not in model_files:
            self.model_var.set(model_files[-1])

        self.load_selected_model()
        self.update_prediction(silent=True)

    def on_model_changed(self, _event: tk.Event) -> None:
        self.load_selected_model()
        self.update_prediction(silent=True)

    def on_toggle_test_all(self) -> None:
        self.update_prediction(silent=True)

    def _get_or_load_model(self, model_name: str) -> SimpleMLP:
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_path = self.models_dir / model_name
        model, _metadata = SimpleMLP.load(model_path, backend="cpu")
        self.loaded_models[model_name] = model
        return model

    def load_selected_model(self) -> None:
        selected = self.model_var.get().strip()
        if not selected:
            self.model = None
            self.model_name = None
            return

        try:
            model = self._get_or_load_model(selected)
            self.model = model
            self.model_name = selected
            self.result_var.set(
                f"Modell geladen: {selected}\n"
                f"Backend: cpu\n"
                f"Ordner: {self.models_dir}"
            )
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.model_name = None
            messagebox.showerror("Fehler", f"Modell konnte nicht geladen werden:\n{exc}")

    def on_draw(self, event: tk.Event) -> None:
        x = int(event.x)
        y = int(event.y)
        self._paint_grid(x, y, direction=1.0)

    def on_erase(self, event: tk.Event) -> None:
        x = int(event.x)
        y = int(event.y)
        self._paint_grid(x, y, direction=-1.0)

    def _paint_grid(self, x: int, y: int, direction: float) -> None:
        gx = x // self.display_scale
        gy = y // self.display_scale
        radius = self.brush_radius
        radius_sq = max(1, radius * radius)
        for dy in range(-self.brush_radius, self.brush_radius + 1):
            for dx in range(-self.brush_radius, self.brush_radius + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq > radius_sq:
                    continue
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    falloff = 1.0 - (dist_sq / radius_sq)
                    delta = direction * self.brush_strength * max(0.2, falloff)
                    new_value = float(np.clip(self.grid[ny, nx] + delta, 0.0, 1.0))
                    self.grid[ny, nx] = new_value
                    gray = int(round(new_value * 255.0))
                    color = f"#{gray:02x}{gray:02x}{gray:02x}"
                    self.canvas.itemconfig(self.pixel_ids[ny][nx], fill=color)
        self.update_prediction(silent=True)

    def clear_canvas(self) -> None:
        self.grid.fill(0.0)
        self.canvas.itemconfig("pixel", fill="black")
        self.result_var.set("Zeichnung geloescht. Zeichne eine Ziffer.")
        self.all_models_var.set("Zeichnung leer.")

    def _update_all_models_predictions(self, x_input: np.ndarray) -> None:
        model_files = list(self.model_combo["values"])
        if not model_files:
            self.all_models_var.set("Keine Modelle gefunden.")
            return

        lines = []
        for model_name in model_files:
            try:
                model = self._get_or_load_model(model_name)
                x_device = model.asarray(x_input, dtype=model.xp.float32)
                probs = model.to_numpy(model.predict_proba(x_device)[0])
                pred = int(np.argmax(probs))
                lines.append(f"{model_name}: {pred}")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"{model_name}: Fehler ({exc})")

        self.all_models_var.set("\n".join(lines))

    def update_prediction(self, silent: bool = True) -> None:
        if self.model is None or self.model_name != self.model_var.get().strip():
            self.load_selected_model()
        if self.model is None:
            if not silent:
                messagebox.showwarning("Hinweis", "Bitte zuerst ein Modell laden.")
            return
        if float(np.sum(self.grid)) == 0.0:
            self.result_var.set("Zeichnung leer. Zeichne eine Ziffer.")
            self.all_models_var.set("Zeichnung leer.")
            return

        x_input = self.grid.reshape(1, -1).astype(np.float32)
        if self.test_all_var.get():
            self.result_var.set("Test all ist aktiv.\nEinzelmetriken vom ausgewaehlten Modell sind ausgeblendet.")
            self._update_all_models_predictions(x_input)
            return

        x_device = self.model.asarray(x_input, dtype=self.model.xp.float32)
        probs = self.model.to_numpy(self.model.predict_proba(x_device)[0])
        pred = int(np.argmax(probs))

        lines = [f"Vorhersage: {pred}", "Wahrscheinlichkeiten pro Zahl:"]
        for idx in range(10):
            lines.append(f"  {idx}: {probs[idx] * 100:.2f}%")
        self.result_var.set("\n".join(lines))
        self.all_models_var.set("Test all ist aus.")


def main() -> None:
    root = tk.Tk()
    models_dir = MODELS_DIR
    DigitApp(root, models_dir=models_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
