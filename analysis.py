from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def extract_numeric_matches(path: Path, phrase: str, max_matches: int = 30) -> list[dict]:
    query = phrase.lower().strip()
    matches = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_number, line in enumerate(lines, 1):
        if query in line.lower():
            numbers = [float(x) for x in NUMBER_PATTERN.findall(line)]
            matches.append({
                "line": line_number,
                "text": line.strip(),
                "numbers": numbers,
            })
            if len(matches) >= max_matches:
                break
    return matches


def compute_statistics(values: list[float]) -> dict:
    if not values:
        raise ValueError("values must not be empty")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std_population": math.sqrt(variance),
    }


def convergence_report(x: list[float], y: list[float], tolerance: float) -> dict:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have the same length and contain at least two points")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    differences = [abs(y[i] - y[i - 1]) for i in range(1, len(y))]
    converged_from = None
    for index in range(len(y)):
        remaining = differences[index:] if index < len(differences) else []
        if remaining and all(delta <= tolerance for delta in remaining):
            converged_from = x[index]
            break

    return {
        "x": x,
        "y": y,
        "successive_absolute_differences": differences,
        "tolerance": tolerance,
        "converged_from_x": converged_from,
    }


def save_xy_plot(
    x: list[float],
    y: list[float],
    output_path: Path,
    x_label: str,
    y_label: str,
    title: str,
) -> str:
    if len(x) != len(y) or not x:
        raise ValueError("x and y must have the same non-zero length")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return str(output_path)
