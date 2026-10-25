from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


TEXT_EXTENSIONS = {
    ".inpt", ".ion", ".out", ".refout", ".static", ".refstatic", ".geopt", ".cellopt",
    ".aimd", ".restart", ".m", ".txt", ".md", ".json", ".csv", ".dat", ".tex", ".bib", ".rst"
}
DIRECTIVE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
PSP_PATH_PATTERN = re.compile(r"(?P<path>[^\s#]+\.psp8)", re.IGNORECASE)
ION_BLOCK_DIRECTIVES = {"ATOM_TYPE", "PSEUDO_POT", "N_TYPE_ATOM", "COORD", "COORD_FRAC", "RELAX", "SPIN", "ATOMIC_MASS"}
OUTPUT_SUFFIXES = (".out", ".static", ".geopt", ".cellopt", ".aimd", ".restart", ".refout", ".refstatic")


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, relative_path: str | Path) -> Path:
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("Path escapes the configured workspace.")
        return candidate

    def list_files(self, relative_dir: str = ".", extension: str = "") -> list[str]:
        folder = self.resolve(relative_dir)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Directory not found: {relative_dir}")
        wanted = extension.lower().strip()
        files = []
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if wanted and path.suffix.lower() != wanted:
                continue
            files.append(str(path.relative_to(self.root)))
        return sorted(files)

    def discover_cases(self, relative_dir: str = ".") -> list[dict]:
        folder = self.resolve(relative_dir)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Directory not found: {relative_dir}")
        cases = []
        for inpt in sorted(folder.rglob("*.inpt")):
            if not inpt.is_file():
                continue
            stem = inpt.with_suffix("")
            ion = Path(str(stem) + ".ion")
            relative_stem = str(stem.relative_to(self.root))
            cases.append({
                "case_stem": relative_stem,
                "inpt": str(inpt.relative_to(self.root)),
                "ion": str(ion.relative_to(self.root)) if ion.exists() else None,
                "valid_pair": ion.exists(),
                "outputs": self.output_files(relative_stem),
            })
        return cases

    def read_text(self, relative_path: str, start_line: int = 1, end_line: int = 250) -> str:
        path = self.resolve(relative_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {relative_path}")
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError(f"Refusing to read non-text file as text: {relative_path}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return ""
        start = max(1, start_line)
        end = min(len(lines), max(start, end_line))
        return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))

    def write_text(self, relative_path: str, content: str, overwrite: bool = False) -> str:
        path = self.resolve(relative_path)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError(f"Unsupported text file type: {path.suffix}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {relative_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(self.root))

    def search_text(self, query: str, relative_dir: str = ".", max_matches: int = 30) -> list[dict]:
        folder = self.resolve(relative_dir)
        query_lower = query.lower()
        matches = []
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                if query_lower in line.lower():
                    matches.append({"file": str(path.relative_to(self.root)), "line": line_number, "text": line.strip()})
                    if len(matches) >= max_matches:
                        return matches
        return matches

    def case_files(self, case_stem: str) -> dict:
        stem = self.resolve(case_stem)
        inpt = Path(str(stem) + ".inpt")
        ion = Path(str(stem) + ".ion")
        return {
            "stem": str(stem.relative_to(self.root)),
            "inpt": str(inpt.relative_to(self.root)) if inpt.exists() else None,
            "ion": str(ion.relative_to(self.root)) if ion.exists() else None,
            "valid_pair": inpt.exists() and ion.exists(),
        }

    def read_directives(self, relative_path: str) -> list[dict]:
        path = self.resolve(relative_path)
        directives = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = DIRECTIVE_PATTERN.match(line)
            if match:
                directives.append({"line": line_number, "parameter": match.group(1), "value": match.group(2)})
        return directives

    def read_inpt_parameters(self, case_stem: str) -> list[dict]:
        stem = self.resolve(case_stem)
        path = Path(str(stem) + ".inpt")
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return self.read_directives(str(path.relative_to(self.root)))

    def parse_ion_blocks(self, case_stem: str) -> list[dict]:
        stem = self.resolve(case_stem)
        ion = Path(str(stem) + ".ion")
        if not ion.exists():
            raise FileNotFoundError(f"Ion file not found: {ion}")
        lines = ion.read_text(encoding="utf-8", errors="replace").splitlines()
        blocks: list[dict] = []
        current: dict | None = None
        active_list: str | None = None
        for line_number, line in enumerate(lines, 1):
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            match = DIRECTIVE_PATTERN.match(stripped)
            if match:
                key = match.group(1).upper()
                value = match.group(2).strip()
                if key == "ATOM_TYPE":
                    if current:
                        blocks.append(current)
                    current = {"atom_type": value.split()[0] if value else "", "line": line_number}
                    active_list = None
                    continue
                if current is None:
                    current = {"atom_type": "", "line": line_number}
                if key == "PSEUDO_POT":
                    current["pseudopotential"] = value.split()[0] if value else ""
                    active_list = None
                elif key == "N_TYPE_ATOM":
                    try:
                        current["n_type_atom"] = int(value.split()[0])
                    except Exception:
                        current["n_type_atom"] = value
                    active_list = None
                elif key in {"COORD", "COORD_FRAC", "RELAX", "SPIN"}:
                    active_list = key.lower()
                    current.setdefault(active_list, [])
                elif key == "ATOMIC_MASS":
                    current["atomic_mass"] = value
                    active_list = None
                else:
                    active_list = None
                continue
            if current is not None and active_list:
                values = stripped.split()
                if active_list == "spin":
                    current[active_list].append(values[0])
                else:
                    current[active_list].append(values)
        if current:
            blocks.append(current)
        return blocks

    def set_inpt_parameter(self, case_stem: str, parameter: str, value: str, allow_append: bool = True) -> dict:
        stem = self.resolve(case_stem)
        path = Path(str(stem) + ".inpt")
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        pattern = re.compile(rf"^(\s*){re.escape(parameter)}\s*:", re.IGNORECASE)
        changed_line = None
        old_line = None
        for index, line in enumerate(lines):
            match = pattern.match(line)
            if match:
                old_line = line
                lines[index] = f"{match.group(1)}{parameter}: {value}"
                changed_line = index + 1
                break
        if changed_line is None:
            if not allow_append:
                raise ValueError(f"Parameter {parameter} is not present and appending was not allowed.")
            lines.append(f"{parameter}: {value}")
            changed_line = len(lines)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"path": str(path.relative_to(self.root)), "parameter": parameter, "value": value, "line": changed_line, "previous": old_line, "appended": old_line is None}

    def remove_inpt_parameter(self, case_stem: str, parameter: str) -> dict:
        stem = self.resolve(case_stem)
        path = Path(str(stem) + ".inpt")
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        pattern = re.compile(rf"^\s*{re.escape(parameter)}\s*:", re.IGNORECASE)
        kept = []
        removed = []
        for line in lines:
            if pattern.match(line):
                removed.append(line)
            else:
                kept.append(line)
        if removed:
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return {"path": str(path.relative_to(self.root)), "parameter": parameter, "removed": removed, "count": len(removed)}

    def _rewrite_psp_paths(self, source_ion: Path, destination_ion: Path) -> list[dict]:
        lines = destination_ion.read_text(encoding="utf-8", errors="replace").splitlines()
        changes = []
        updated_lines = []
        for line_number, line in enumerate(lines, 1):
            updated = line
            for match in list(PSP_PATH_PATTERN.finditer(line)):
                raw_path = match.group("path").strip('"\'')
                raw_obj = Path(raw_path)
                source_target = raw_obj.resolve() if raw_obj.is_absolute() else (source_ion.parent / raw_obj).resolve()
                if not source_target.exists():
                    continue
                new_relative = os.path.relpath(source_target, destination_ion.parent).replace("\\", "/")
                updated = updated.replace(match.group("path"), new_relative)
                changes.append({"line": line_number, "old": match.group("path"), "new": new_relative, "target_exists": True})
            updated_lines.append(updated)
        destination_ion.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        return changes

    def duplicate_case(self, source_stem: str, destination_stem: str, reuse_existing: bool = False) -> dict:
        source = self.resolve(source_stem)
        destination = self.resolve(destination_stem)
        source_inpt = Path(str(source) + ".inpt")
        source_ion = Path(str(source) + ".ion")
        if not source_inpt.exists() or not source_ion.exists():
            raise FileNotFoundError("Source case must contain both .inpt and .ion files with the same stem.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_inpt = Path(str(destination) + ".inpt")
        destination_ion = Path(str(destination) + ".ion")
        if destination_inpt.exists() or destination_ion.exists():
            if reuse_existing and destination_inpt.exists() and destination_ion.exists():
                refs = self.pseudopotential_references(destination_stem)
                changes = self._rewrite_psp_paths(source_ion, destination_ion) if any(not ref["exists"] for ref in refs) else []
                return {"source": source_stem, "destination": destination_stem, "files": [str(destination_inpt.relative_to(self.root)), str(destination_ion.relative_to(self.root))], "reused_existing": True, "pseudopotential_path_changes": changes}
            raise FileExistsError(f"Destination case already exists: {destination_stem}")
        shutil.copy2(source_inpt, destination_inpt)
        shutil.copy2(source_ion, destination_ion)
        changes = self._rewrite_psp_paths(source_ion, destination_ion)
        return {"source": source_stem, "destination": destination_stem, "files": [str(destination_inpt.relative_to(self.root)), str(destination_ion.relative_to(self.root))], "reused_existing": False, "pseudopotential_path_changes": changes}

    def copy_external_case(self, source_stem: Path, destination_stem: str) -> dict:
        source_stem = source_stem.resolve()
        source_inpt = Path(str(source_stem) + ".inpt")
        source_ion = Path(str(source_stem) + ".ion")
        if not source_inpt.exists() or not source_ion.exists():
            raise FileNotFoundError("Reference case must contain matching .inpt and .ion files.")
        destination = self.resolve(destination_stem)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_inpt = Path(str(destination) + ".inpt")
        destination_ion = Path(str(destination) + ".ion")
        if destination_inpt.exists() or destination_ion.exists():
            raise FileExistsError(f"Destination case already exists: {destination_stem}")
        shutil.copy2(source_inpt, destination_inpt)
        shutil.copy2(source_ion, destination_ion)
        changes = self._rewrite_psp_paths(source_ion, destination_ion)
        return {"destination": destination_stem, "files": [str(destination_inpt.relative_to(self.root)), str(destination_ion.relative_to(self.root))], "pseudopotential_path_changes": changes}

    def pseudopotential_references(self, case_stem: str) -> list[dict]:
        stem = self.resolve(case_stem)
        ion = Path(str(stem) + ".ion")
        if not ion.exists():
            return []
        refs = []
        for line_number, line in enumerate(ion.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for match in PSP_PATH_PATTERN.finditer(line):
                raw = match.group("path").strip('"\'')
                obj = Path(raw)
                resolved = obj.resolve() if obj.is_absolute() else (ion.parent / obj).resolve()
                refs.append({"line": line_number, "path": raw, "resolved": str(resolved), "exists": resolved.exists()})
        return refs

    def output_files(self, case_stem: str) -> list[str]:
        stem = self.resolve(case_stem)
        matches: set[str] = set()
        for path in stem.parent.glob(stem.name + ".*"):
            if path.is_file() and any(path.name.startswith(stem.name + suffix) for suffix in OUTPUT_SUFFIXES):
                matches.add(str(path.relative_to(self.root)))

        # M-SPARC allows OUTPUT_FILE to override the normal <case>.out name.
        inpt = Path(str(stem) + ".inpt")
        if inpt.exists():
            for item in self.read_directives(str(inpt.relative_to(self.root))):
                if item["parameter"].upper() == "OUTPUT_FILE" and item["value"].strip():
                    configured = (stem.parent / item["value"].strip().split()[0]).resolve()
                    if configured.exists() and configured.is_file() and (configured == self.root or self.root in configured.parents):
                        matches.add(str(configured.relative_to(self.root)))
                    break
        return sorted(matches)

    def newest_output_for_case(self, case_stem: str) -> str | None:
        files = [self.resolve(path) for path in self.output_files(case_stem)]
        if not files:
            return None
        newest = max(files, key=lambda path: path.stat().st_mtime)
        return str(newest.relative_to(self.root))


def safe_value_token(value: object) -> str:
    token = str(value).strip().replace("-", "m").replace("+", "p").replace(".", "p")
    token = re.sub(r"[^A-Za-z0-9_]+", "_", token)
    return token.strip("_") or "value"
