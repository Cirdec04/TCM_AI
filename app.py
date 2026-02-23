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
        self.root.title("KNN Ziffernerkennung")
        self.root.resizable(False, False)

        self.models_dir = models_dir
        self.grid_size = 28
        self.display_scale = 12
        self.canvas_size = self.grid_size * self.display_scale
        self.brush_radius = 1

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.pixel_ids: list[list[int]] = []
        self.model: SimpleMLP | None = None
        self.model_name: str | None = None

        self.model_var = tk.StringVar()
        self.result_var = tk.StringVar(value="Noch keine Vorhersage.")

        self._build_ui()
        self.refresh_model_list()

    def _build_ui(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Modell:").pack(side="left")
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, state="readonly", width=30)
        self.model_combo.pack(side="left", padx=8)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)

        ttk.Button(top_frame, text="Modelle neu laden", command=self.refresh_model_list).pack(side="left")

        canvas_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        canvas_frame.pack()

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=1)
        self.canvas.pack()
        self._init_pixel_canvas()
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<Button-1>", self.on_draw)
        self.canvas.bind("<B3-Motion>", self.on_erase)
        self.canvas.bind("<Button-3>", self.on_erase)
        ttk.Label(
            canvas_frame,
            text=f"Zoom x{self.display_scale} (28x28 Raster) | Links: zeichnen | Rechts: radieren",
        ).pack(pady=(6, 0))

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
            self.result_var.set(f"Keine Modelle gefunden in:\n{self.models_dir}")
            return

        if self.model_var.get() not in model_files:
            self.model_var.set(model_files[-1])

        self.load_selected_model()

    def on_model_changed(self, _event: tk.Event) -> None:
        self.load_selected_model()
        self.update_prediction(silent=True)

    def load_selected_model(self) -> None:
        selected = self.model_var.get().strip()
        if not selected:
            self.model = None
            self.model_name = None
            return

        model_path = self.models_dir / selected
        try:
            model, metadata = SimpleMLP.load(model_path)
            self.model = model
            self.model_name = selected
            size_info = metadata.get("size", "unbekannt")
            self.result_var.set(
                f"Modell geladen: {selected} (size: {size_info})\n"
                f"Ordner: {self.models_dir}"
            )
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.model_name = None
            messagebox.showerror("Fehler", f"Modell konnte nicht geladen werden:\n{exc}")

    def on_draw(self, event: tk.Event) -> None:
        x = int(event.x)
        y = int(event.y)
        self._paint_grid(x, y, value=1.0, color="white")

    def on_erase(self, event: tk.Event) -> None:
        x = int(event.x)
        y = int(event.y)
        self._paint_grid(x, y, value=0.0, color="black")

    def _paint_grid(self, x: int, y: int, value: float, color: str) -> None:
        gx = x // self.display_scale
        gy = y // self.display_scale
        for dy in range(-self.brush_radius, self.brush_radius + 1):
            for dx in range(-self.brush_radius, self.brush_radius + 1):
                if dx * dx + dy * dy > self.brush_radius * self.brush_radius:
                    continue
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.grid[ny, nx] = value
                    self.canvas.itemconfig(self.pixel_ids[ny][nx], fill=color)
        self.update_prediction(silent=True)

    def clear_canvas(self) -> None:
        self.grid.fill(0.0)
        self.canvas.itemconfig("pixel", fill="black")
        self.result_var.set("Zeichnung geloescht. Zeichne eine Ziffer.")

    def update_prediction(self, silent: bool = True) -> None:
        if self.model is None or self.model_name != self.model_var.get().strip():
            self.load_selected_model()
        if self.model is None:
            if not silent:
                messagebox.showwarning("Hinweis", "Bitte zuerst ein Modell laden.")
            return
        if float(np.sum(self.grid)) == 0.0:
            self.result_var.set("Zeichnung leer. Zeichne eine Ziffer.")
            return

        x_input = self.grid.reshape(1, -1).astype(np.float32)
        probs = self.model.predict_proba(x_input)[0]
        pred = int(np.argmax(probs))

        lines = [f"Vorhersage: {pred}", "Wahrscheinlichkeiten pro Zahl (0-9):"]
        for idx in range(10):
            lines.append(f"  {idx}: {probs[idx] * 100:.2f}%")
        self.result_var.set("\n".join(lines))


def main() -> None:
    root = tk.Tk()
    models_dir = MODELS_DIR
    app = DigitApp(root, models_dir=models_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
