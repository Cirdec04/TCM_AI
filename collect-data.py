from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from deps import ensure_requirements_installed

ensure_requirements_installed(required_modules=("numpy", "PIL"))

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CUSTOM_DATA_DIR = DATA_DIR / "custom"
CUSTOM_TRAIN_DIR = CUSTOM_DATA_DIR / "training"
CUSTOM_TEST_DIR = CUSTOM_DATA_DIR / "testing"


class DataCollectorUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TCM Custom Data Collector")
        self.root.resizable(False, False)

        self.grid_size = 28
        self.display_scale = 12
        self.canvas_size = self.grid_size * self.display_scale
        self.brush_radius = 2
        self.brush_strength = 0.35

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.pixel_ids: list[list[int]] = []
        self.rng = np.random.default_rng()

        self.digit_var = tk.StringVar(value="0")
        self.train_ratio_var = tk.DoubleVar(value=0.8)
        self.status_var = tk.StringVar(value="Bereit")
        self.counts_var = tk.StringVar(value="")

        self._ensure_dirs()
        self._build_ui()
        self._refresh_counts()

    def _ensure_dirs(self) -> None:
        for split_root in (CUSTOM_TRAIN_DIR, CUSTOM_TEST_DIR):
            for digit in range(10):
                (split_root / str(digit)).mkdir(parents=True, exist_ok=True)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        collector_tab = ttk.Frame(notebook, padding=0)
        stats_tab = ttk.Frame(notebook, padding=10)
        notebook.add(collector_tab, text="Sammeln")
        notebook.add(stats_tab, text="Stats")

        top = ttk.Frame(collector_tab, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Label:").pack(side="left")
        digit_combo = ttk.Combobox(top, width=6, state="readonly", textvariable=self.digit_var, values=[str(i) for i in range(10)])
        digit_combo.pack(side="left", padx=(6, 12))
        digit_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_counts())

        ttk.Label(top, text="Train-Anteil:").pack(side="left")
        ratio_spin = ttk.Spinbox(
            top,
            from_=0.5,
            to=0.99,
            increment=0.01,
            width=6,
            textvariable=self.train_ratio_var,
            format="%.2f",
        )
        ratio_spin.pack(side="left", padx=(6, 0))

        canvas_frame = ttk.Frame(collector_tab, padding=(10, 0, 10, 10))
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
            text=f"28x28 Raster (Zoom x{self.display_scale}) | Links: zeichnen | Rechts: radieren",
        ).pack(pady=(6, 0))

        buttons = ttk.Frame(collector_tab, padding=(10, 0, 10, 0))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Speichern", command=self.save_sample).pack(side="left")
        ttk.Button(buttons, text="Loeschen", command=self.clear_canvas).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Counts aktualisieren", command=self._refresh_counts).pack(side="left", padx=(8, 0))

        info = ttk.Frame(collector_tab, padding=(10, 8, 10, 10))
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.status_var, justify="left").pack(anchor="w")

        ttk.Label(
            stats_tab,
            text="Custom-Dataset-Statistiken",
            font=("TkDefaultFont", 10, "bold"),
            justify="left",
        ).pack(anchor="w")
        ttk.Label(stats_tab, textvariable=self.counts_var, justify="left").pack(anchor="w", pady=(8, 0))

        self.root.bind("<Return>", lambda _e: self.save_sample())
        self.root.bind("<Escape>", lambda _e: self.clear_canvas())

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

    def on_draw(self, event: tk.Event) -> None:
        self._paint_grid(int(event.x), int(event.y), direction=1.0)

    def on_erase(self, event: tk.Event) -> None:
        self._paint_grid(int(event.x), int(event.y), direction=-1.0)

    def _paint_grid(self, x: int, y: int, direction: float) -> None:
        gx = x // self.display_scale
        gy = y // self.display_scale
        radius_sq = max(1, self.brush_radius * self.brush_radius)
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

    def clear_canvas(self, silent: bool = False) -> None:
        self.grid.fill(0.0)
        self.canvas.itemconfig("pixel", fill="black")
        if not silent:
            self.status_var.set("Canvas geloescht.")

    def _count_split_digit(self, split_root: Path, digit: int) -> int:
        digit_dir = split_root / str(digit)
        if not digit_dir.exists():
            return 0
        return len([p for p in digit_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])

    def _refresh_counts(self) -> None:
        try:
            digit = int(self.digit_var.get())
        except ValueError:
            digit = 0
            self.digit_var.set("0")

        per_digit_lines: list[str] = []
        train_total = 0
        test_total = 0
        selected_train = 0
        selected_test = 0

        for current_digit in range(10):
            train_count = self._count_split_digit(CUSTOM_TRAIN_DIR, current_digit)
            test_count = self._count_split_digit(CUSTOM_TEST_DIR, current_digit)
            digit_total = train_count + test_count
            per_digit_lines.append(
                f"{current_digit}: train={train_count}, test={test_count}, total={digit_total}"
            )
            train_total += train_count
            test_total += test_count
            if current_digit == digit:
                selected_train = train_count
                selected_test = test_count

        total = train_total + test_total
        self.counts_var.set(
            f"Ausgewaehlt ({digit}): train={selected_train}, test={selected_test}, total={selected_train + selected_test}\n"
            f"Gesamt custom: train={train_total}, test={test_total}, total={total}\n\n"
            f"Pro Ziffer:\n" + "\n".join(per_digit_lines) + "\n\n"
            f"Pfad: {CUSTOM_DATA_DIR}"
        )

    def save_sample(self) -> None:
        if float(np.sum(self.grid)) <= 0.0:
            messagebox.showwarning("Leer", "Bitte zuerst eine Ziffer zeichnen.")
            return

        try:
            digit = int(self.digit_var.get())
        except ValueError:
            messagebox.showerror("Fehler", "Ungueltiges Label. Erlaubt sind 0-9.")
            return
        if digit < 0 or digit > 9:
            messagebox.showerror("Fehler", "Ungueltiges Label. Erlaubt sind 0-9.")
            return

        train_ratio = float(np.clip(self.train_ratio_var.get(), 0.01, 0.99))
        split = "training" if float(self.rng.random()) < train_ratio else "testing"
        split_root = CUSTOM_TRAIN_DIR if split == "training" else CUSTOM_TEST_DIR
        target_dir = split_root / str(digit)
        target_dir.mkdir(parents=True, exist_ok=True)

        stamp = int(time.time() * 1000)
        nonce = int(self.rng.integers(0, 10_000_000))
        file_path = target_dir / f"{stamp}_{nonce:07d}.png"

        image_u8 = np.clip(self.grid * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(image_u8, mode="L").save(file_path, format="PNG")

        self.status_var.set(
            f"Gespeichert: {file_path.name} -> data/custom/{split}/{digit}/ "
            f"(train_ratio={train_ratio:.2f})"
        )
        self.clear_canvas(silent=True)
        self._refresh_counts()


def main() -> None:
    root = tk.Tk()
    DataCollectorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
