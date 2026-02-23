from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_REQUIRED_MODULES = ("numpy", "matplotlib", "PIL")


def _missing_modules(required_modules: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def ensure_requirements_installed(
    required_modules: Iterable[str] = DEFAULT_REQUIRED_MODULES,
    requirements_file: str = "requirements.txt",
) -> None:
    missing_before = _missing_modules(required_modules)
    if not missing_before:
        return

    root_dir = Path(__file__).resolve().parent
    requirements_path = root_dir / requirements_file
    if not requirements_path.exists():
        raise SystemExit(f"Fehlende Datei: {requirements_path}")

    print(f"Fehlende Pakete erkannt: {', '.join(missing_before)}")
    print(f"Installiere automatisch ueber: {requirements_path}")

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "Abhaengigkeiten konnten nicht installiert werden. "
            "Bitte `pip install -r requirements.txt` manuell ausfuehren."
        )

    missing_after = _missing_modules(required_modules)
    if missing_after:
        raise SystemExit(
            "Diese Module fehlen weiterhin: "
            + ", ".join(missing_after)
            + ". Bitte Umgebung/Python-Interpreter pruefen."
        )
