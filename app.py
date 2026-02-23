from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from deps import ensure_requirements_installed

ensure_requirements_installed()

import numpy as np

from nn import SimpleMLP


class DigitApp:
    def __init__(self, root: tk.Tk, models_dir: Path) -> None:
        self.root = root
        self.root.title("KNN Ziffernerkennung")
        self.root.resizable(False, False)

        self.models_dir = models_dir
        self.canvas_size = 280
        self.grid_size = 28
        self.scale = self.canvas_size // self.grid_size
        self.brush_radius = 2

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
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

        ttk.Button(top_frame, text="Modelle neu laden", command=self.refresh_model_list).pack(side="left")

        canvas_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        canvas_frame.pack()

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=1)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<Button-1>", self.on_draw)

        buttons = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")

        ttk.Button(buttons, text="Vorhersage", command=self.predict).pack(side="left")
        ttk.Button(buttons, text="Loeschen", command=self.clear_canvas).pack(side="left", padx=8)

        result_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        result_frame.pack(fill="x")
        ttk.Label(result_frame, textvariable=self.result_var, justify="left").pack(anchor="w")

    def refresh_model_list(self) -> None:
        model_files = sorted([p.name for p in self.models_dir.glob("TCM-o*.npz")])
        self.model_combo["values"] = model_files

        if not model_files:
            self.model_var.set("")
            self.model = None
            self.model_name = None
            self.result_var.set("Keine Modelle gefunden in ./models")
            return

        if self.model_var.get() not in model_files:
            self.model_var.set(model_files[-1])

        self.load_selected_model()

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
            self.result_var.set(f"Modell geladen: {selected} (size: {size_info})")
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.model_name = None
            messagebox.showerror("Fehler", f"Modell konnte nicht geladen werden:\n{exc}")

    def on_draw(self, event: tk.Event) -> None:
        x = int(event.x)
        y = int(event.y)
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self._paint_grid(x, y)

    def _paint_grid(self, x: int, y: int) -> None:
        gx = int(x / self.scale)
        gy = int(y / self.scale)
        for dy in range(-self.brush_radius, self.brush_radius + 1):
            for dx in range(-self.brush_radius, self.brush_radius + 1):
                if dx * dx + dy * dy > self.brush_radius * self.brush_radius:
                    continue
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.grid[ny, nx] = 1.0

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.canvas.configure(bg="black")
        self.grid.fill(0.0)
        self.result_var.set("Zeichnung geloescht.")

    def predict(self) -> None:
        if self.model is None or self.model_name != self.model_var.get().strip():
            self.load_selected_model()
        if self.model is None:
            messagebox.showwarning("Hinweis", "Bitte zuerst ein Modell laden.")
            return
        if float(np.sum(self.grid)) == 0.0:
            messagebox.showinfo("Hinweis", "Bitte zuerst eine Ziffer zeichnen.")
            return

        x_input = self.grid.reshape(1, -1).astype(np.float32)
        probs = self.model.predict_proba(x_input)[0]
        pred = int(np.argmax(probs))

        top3 = np.argsort(probs)[-3:][::-1]
        lines = [f"Vorhersage: {pred}", "Top 3 Wahrscheinlichkeiten:"]
        for idx in top3:
            lines.append(f"  {idx}: {probs[idx] * 100:.2f}%")
        self.result_var.set("\n".join(lines))


def main() -> None:
    root = tk.Tk()
    models_dir = Path("models")
    app = DigitApp(root, models_dir=models_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
