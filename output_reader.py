from __future__ import annotations

import re
from pathlib import Path

from analysis import NUMBER_PATTERN
from workspace import Workspace


QUANTITY_LABELS = {
    "total energy": ["Total free energy", "Total energy"],
    "total free energy": ["Total free energy"],
    "free energy": ["Total free energy", "Free energy per atom"],
    "free energy per atom": ["Free energy per atom"],
    "fermi level": ["Fermi level"],
    "fermi energy": ["Fermi level"],
    "maximum force": ["Maximum force"],
    "max force": ["Maximum force"],
    "average force": ["Average force"],
    "pressure": ["Pressure"],
    "maximum stress": ["Maximum stress"],
    "max stress": ["Maximum stress"],
    "magnetization": ["Net Magnetization", "Magnetization"],
    "net magnetization": ["Net Magnetization"],
    "scf iterations": ["Total number of SCF"],
    "number of scf": ["Total number of SCF"],
    "walltime": ["Total walltime"],
    "total walltime": ["Total walltime"],
}


def normalize_quantity(quantity: str) -> str:
    return " ".join(quantity.lower().replace("_", " ").replace("-", " ").split())


class MSPARCOutputReader:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def _priority_files(self, case_stem: str) -> list[Path]:
        files = [self.workspace.resolve(path) for path in self.workspace.output_files(case_stem)]
        priority = {".out": 0, ".static": 1, ".geopt": 2, ".cellopt": 3, ".aimd": 4, ".restart": 5, ".refout": 6, ".refstatic": 7}
        return sorted(files, key=lambda path: (priority.get(path.suffix.lower(), 99), path.name))

    def query(self, case_stem: str, quantity: str, max_matches: int = 20) -> dict:
        files = self._priority_files(case_stem)
        if not files:
            return {"success": False, "error": f"No recognized output files found for {case_stem}.", "case": case_stem}

        normalized = normalize_quantity(quantity)
        labels = QUANTITY_LABELS.get(normalized, [quantity])
        matches = []
        for path in files:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line_number, line in enumerate(lines, 1):
                for label in labels:
                    if label.lower() not in line.lower():
                        continue
                    numbers = [float(value) for value in NUMBER_PATTERN.findall(line)]
                    # Prefer numbers after ':' so line labels like SCF#1 do not contaminate results.
                    after_colon = line.split(":", 1)[1] if ":" in line else line
                    after_numbers = [float(value) for value in NUMBER_PATTERN.findall(after_colon)]
                    if after_numbers:
                        numbers = after_numbers
                    matches.append({
                        "file": str(path.relative_to(self.workspace.root)),
                        "line": line_number,
                        "text": line.strip(),
                        "numbers": numbers,
                        "matched_label": label,
                    })
                    break
                    
        if not matches:
            return {
                "success": False,
                "case": case_stem,
                "quantity": quantity,
                "error": f"Could not find {quantity!r} in recognized output files.",
                "searched_files": [str(path.relative_to(self.workspace.root)) for path in files],
            }

        # The final summary value is normally the last occurrence. Return all matches for auditability.
        final = matches[-1]
        return {
            "success": True,
            "case": case_stem,
            "quantity": quantity,
            "value": final["numbers"][0] if final["numbers"] else None,
            "numbers": final["numbers"],
            "source_file": final["file"],
            "source_line": final["line"],
            "source_text": final["text"],
            "matches": matches[-max_matches:],
        }

    def query_many(self, case_stems: list[str], quantity: str) -> dict:
        return {"quantity": quantity, "results": [self.query(case, quantity) for case in case_stems]}
