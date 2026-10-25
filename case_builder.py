from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from reference import MSPARCReference
from workspace import Workspace


GRID_CONTROL = {"MESH_SPACING", "FD_GRID", "ECUT"}
LATTICE_SCALE_CONTROL = {"CELL", "LATVEC_SCALE"}
CANONICAL_INPT_ORDER = [
    "CELL", "LATVEC", "LATVEC_SCALE", "MESH_SPACING", "FD_GRID", "ECUT", "FD_ORDER", "BC",
    "EXCHANGE_CORRELATION", "SPIN_TYP", "KPOINT_GRID", "KPOINT_SHIFT", "ELEC_TEMP_TYPE", "ELEC_TEMP",
    "SMEARING", "NSTATES", "TOL_SCF", "MAXIT_SCF", "TOL_POISSON", "TOL_PSEUDOCHARGE",
    "MIXING_VARIABLE", "MIXING_PRECOND", "MIXING_PARAMETER", "PRINT_FORCES", "PRINT_ATOMS", "CALC_STRESS",
]


def _format_number(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.15g}"
    return str(value).strip()


def _format_row(values) -> str:
    return " ".join(_format_number(value) for value in values)


def _normalize_parameters(raw: dict) -> dict[str, object]:
    return {str(key).strip().upper(): value for key, value in raw.items()}


class MSPARCCaseBuilder:
    """Write M-SPARC input files from structured scientific data, never from raw LLM text."""

    def __init__(self, workspace: Workspace, reference: MSPARCReference, msparc_root: Path):
        self.workspace = workspace
        self.reference = reference
        self.msparc_root = msparc_root.resolve()

    def _pseudo_for(self, element: str, requested: str | None, destination_dir: Path) -> tuple[str | None, dict]:
        if requested and requested.strip().lower() not in {"auto", "default"}:
            raw = Path(requested.strip())
            candidates = []
            if raw.is_absolute():
                candidates.append(raw)
            else:
                candidates.extend([
                    (self.msparc_root / raw).resolve(),
                    (self.msparc_root / "psps" / raw.name).resolve(),
                    (destination_dir / raw).resolve(),
                ])
            target = next((path for path in candidates if path.exists()), None)
            if target is None:
                return None, {"element": element, "error": f"Pseudopotential not found: {requested}"}
            relative = os.path.relpath(target, destination_dir).replace("\\", "/")
            return relative, {"element": element, "chosen": str(target), "source": "user supplied"}

        result = self.reference.preferred_pseudopotential(element)
        if not result.get("found"):
            return None, {
                "element": element,
                "error": "Could not choose one pseudopotential unambiguously from local M-SPARC evidence.",
                "candidates": result.get("candidates", []),
            }
        target = Path(result["chosen"]).resolve()
        relative = os.path.relpath(target, destination_dir).replace("\\", "/")
        return relative, result

    def _normalize_species(self, spec: dict) -> tuple[list[dict], list[str]]:
        species = spec.get("species")
        errors: list[str] = []
        if species is None and isinstance(spec.get("atoms"), list):
            # Accept a flat atom list: [{element, position/coordinates, ...}, ...]
            groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
            for atom in spec["atoms"]:
                if not isinstance(atom, dict) or not atom.get("element"):
                    errors.append("Every atom entry must contain an element.")
                    continue
                mode = str(atom.get("coordinate_mode") or spec.get("coordinate_mode") or "").lower()
                pseudo = str(atom.get("pseudopotential") or "auto")
                groups[(str(atom["element"]).capitalize(), mode, pseudo)].append(atom)
            species = []
            for (element, mode, pseudo), atoms in groups.items():
                species.append({
                    "element": element,
                    "coordinate_mode": mode,
                    "pseudopotential": pseudo,
                    "coordinates": [a.get("position", a.get("coordinates")) for a in atoms],
                    "spin": [a.get("spin") for a in atoms] if any(a.get("spin") is not None for a in atoms) else None,
                    "relax": [a.get("relax") for a in atoms] if any(a.get("relax") is not None for a in atoms) else None,
                })

        if not isinstance(species, list) or not species:
            errors.append("At least one species/atom is required for the .ion file.")
            return [], errors

        normalized = []
        for item in species:
            if not isinstance(item, dict):
                errors.append("Each species must be an object.")
                continue
            element = str(item.get("element") or item.get("atom_type") or "").strip().capitalize()
            coords = item.get("coordinates") or item.get("positions")
            mode = str(item.get("coordinate_mode") or spec.get("coordinate_mode") or "").strip().lower()
            if not element:
                errors.append("Each species needs an element/atom_type.")
                continue
            if mode in {"frac", "fractional", "coord_frac"}:
                mode = "fractional"
            elif mode in {"cart", "cartesian", "coord"}:
                mode = "cartesian"
            else:
                errors.append(f"Species {element}: coordinate_mode must be 'fractional' or 'cartesian'.")
            if not isinstance(coords, list) or not coords:
                errors.append(f"Species {element}: coordinates are required.")
                continue
            for row in coords:
                if not isinstance(row, (list, tuple)) or len(row) != 3:
                    errors.append(f"Species {element}: each coordinate must contain exactly 3 numbers.")
                    break
            spin = item.get("spin")
            relax = item.get("relax")
            if spin is not None and len(spin) != len(coords):
                errors.append(f"Species {element}: SPIN entries must match the number of coordinates.")
            if relax is not None:
                if len(relax) != len(coords) or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in relax):
                    errors.append(f"Species {element}: RELAX must have one 3-value row per atom.")
            normalized.append({
                "element": element,
                "coordinates": coords,
                "coordinate_mode": mode,
                "pseudopotential": item.get("pseudopotential") or "auto",
                "spin": spin,
                "relax": relax,
                "atomic_mass": item.get("atomic_mass"),
            })
        return normalized, errors

    def validate_spec(self, spec: dict) -> dict:
        errors: list[str] = []
        warnings: list[str] = []
        parameters = _normalize_parameters(spec.get("inpt") or spec.get("parameters") or {})
        if not parameters:
            errors.append("The case specification must contain an 'inpt' object with M-SPARC parameters.")

        unknown = [key for key in parameters if not self.reference.known_inpt_keyword(key)]
        if unknown:
            errors.append("Unknown .inpt keyword(s) according to local Manual.tex/tests: " + ", ".join(unknown))

        grid_keys = sorted(GRID_CONTROL & set(parameters))
        if len(grid_keys) != 1:
            errors.append("Specify exactly one of MESH_SPACING, FD_GRID, or ECUT. The M-SPARC manual marks them as mutually exclusive.")

        scale_keys = sorted(LATTICE_SCALE_CONTROL & set(parameters))
        if len(scale_keys) > 1:
            errors.append("CELL and LATVEC_SCALE cannot be specified simultaneously according to the M-SPARC manual.")
        if not ({"CELL", "LATVEC", "LATVEC_SCALE"} & set(parameters)):
            errors.append("Provide the simulation cell using CELL, LATVEC, or LATVEC_SCALE (LATVEC may be combined with either scaling form).")
        if "BC" not in parameters:
            errors.append("BC is required because the M-SPARC manual lists no default boundary condition.")

        if "LATVEC" in parameters:
            matrix = parameters["LATVEC"]
            if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in matrix):
                errors.append("LATVEC must be a 3x3 matrix.")
        if "CELL" in parameters:
            cell = parameters["CELL"]
            if not isinstance(cell, list) or len(cell) != 3:
                errors.append("CELL must contain 3 values.")

        species, species_errors = self._normalize_species(spec)
        errors.extend(species_errors)

        defaults = {}
        for key in ["BC", "FD_ORDER", "EXCHANGE_CORRELATION", "TOL_SCF", "KPOINT_GRID", "KPOINT_SHIFT"]:
            if key not in parameters:
                default = self.reference.default_value(key)
                if default is not None:
                    defaults[key] = default
        if defaults:
            warnings.append("Unspecified optional parameters will use M-SPARC's own defaults; they are not redundantly written into the file.")

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "parameters": parameters,
            "species": species,
            "known_defaults_not_written": defaults,
        }

    def _render_inpt(self, parameters: dict[str, object]) -> str:
        ordered_keys = [key for key in CANONICAL_INPT_ORDER if key in parameters]
        ordered_keys += [key for key in parameters if key not in ordered_keys]
        lines = ["# Generated by M-SPARC Agent from local doc/Manual.tex schema", ""]
        for key in ordered_keys:
            value = parameters[key]
            if key == "LATVEC":
                lines.append("LATVEC:")
                lines.extend(_format_row(row) for row in value)
            elif isinstance(value, list):
                lines.append(f"{key}: {_format_row(value)}")
            else:
                lines.append(f"{key}: {_format_number(value)}")
        return "\n".join(lines).rstrip() + "\n"

    def _render_ion(self, species: list[dict], destination_dir: Path) -> tuple[str | None, list[dict], list[str]]:
        lines = [
            "# Generated by M-SPARC Agent using official M-SPARC .ion syntax",
            "# Each species block follows ATOM_TYPE / PSEUDO_POT / N_TYPE_ATOM / COORD[_FRAC].",
            "",
        ]
        pseudo_info = []
        errors = []
        for item in species:
            element = item["element"]
            pseudo, info = self._pseudo_for(element, item.get("pseudopotential"), destination_dir)
            pseudo_info.append(info)
            if pseudo is None:
                errors.append(info.get("error", f"Could not resolve pseudopotential for {element}."))
                continue
            coords = item["coordinates"]
            lines.append(f"ATOM_TYPE: {element}")
            lines.append(f"PSEUDO_POT: {pseudo}")
            lines.append(f"N_TYPE_ATOM: {len(coords)}")
            if item.get("atomic_mass") is not None:
                lines.append(f"ATOMIC_MASS: {_format_number(item['atomic_mass'])}")
            lines.append("COORD_FRAC:" if item["coordinate_mode"] == "fractional" else "COORD:")
            lines.extend(_format_row(row) for row in coords)
            if item.get("relax") is not None:
                lines.append("RELAX:")
                lines.extend(_format_row(row) for row in item["relax"])
            if item.get("spin") is not None:
                lines.append("SPIN:")
                lines.extend(_format_number(value) for value in item["spin"])
            lines.append("")
        if errors:
            return None, pseudo_info, errors
        return "\n".join(lines).rstrip() + "\n", pseudo_info, []

    def create_case(self, case_stem: str, spec_json: str, overwrite: bool = False) -> dict:
        try:
            spec = json.loads(spec_json)
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"case specification is not valid JSON: {exc}"}
        if not isinstance(spec, dict):
            return {"success": False, "error": "case specification must be a JSON object"}

        validation = self.validate_spec(spec)
        if not validation["valid"]:
            return {"success": False, "error": "Case specification is incomplete or invalid.", "validation": validation}

        stem = self.workspace.resolve(case_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        inpt_path = Path(str(stem) + ".inpt")
        ion_path = Path(str(stem) + ".ion")
        if (inpt_path.exists() or ion_path.exists()) and not overwrite:
            return {"success": False, "error": f"Case already exists: {case_stem}. Set overwrite only when replacement was explicitly requested."}

        inpt_text = self._render_inpt(validation["parameters"])
        ion_text, pseudo_info, ion_errors = self._render_ion(validation["species"], stem.parent)
        if ion_errors:
            return {"success": False, "error": "Could not resolve pseudopotential(s).", "details": ion_errors, "pseudopotentials": pseudo_info}

        inpt_path.write_text(inpt_text, encoding="utf-8")
        ion_path.write_text(ion_text, encoding="utf-8")
        return {
            "success": True,
            "case_stem": str(stem.relative_to(self.workspace.root)),
            "inpt": str(inpt_path.relative_to(self.workspace.root)),
            "ion": str(ion_path.relative_to(self.workspace.root)),
            "pseudopotentials": pseudo_info,
            "manual_defaults_not_written": validation["known_defaults_not_written"],
            "note": "Syntax was rendered locally from structured data; the LLM did not write raw .inpt/.ion text.",
        }
