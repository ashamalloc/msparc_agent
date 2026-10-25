from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REFERENCE_TEXT_EXTENSIONS = {
    ".inpt", ".ion", ".txt", ".md", ".rst", ".tex", ".json", ".csv", ".dat"
}
DIRECTIVE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
FRAME_PATTERN = re.compile(r"\\begin\{frame\}.*?\\end\{frame\}", re.DOTALL)
LABEL_PATTERN = re.compile(r"\\label\{([A-Z][A-Z0-9_]*)\}")
BLOCK_PATTERN = re.compile(
    r"\\begin\{block\}\{([^}]*)\}(.*?)\\end\{block\}", re.DOTALL
)
ATOM_TYPE_PATTERN = re.compile(r"^\s*ATOM_TYPE\s*:\s*([^\s#]+)", re.IGNORECASE)
PSEUDO_PATTERN = re.compile(r"^\s*PSEUDO_POT\s*:\s*([^\s#]+\.psp8)", re.IGNORECASE)


# High-frequency human phrases. The exact keyword is still validated against the local manual.
ION_ONLY_KEYWORDS = {
    "ATOM_TYPE", "PSEUDO_POT", "N_TYPE_ATOM", "COORD", "COORD_FRAC", "RELAX", "SPIN", "ATOMIC_MASS",
    "HUBBARD", "U_ATOM_TYPE", "U_VAL", "CORE_FLAG"
}

COMMON_ALIASES = {
    "mesh spacing": "MESH_SPACING",
    "grid spacing": "MESH_SPACING",
    "finite difference grid": "FD_GRID",
    "fd grid": "FD_GRID",
    "energy cutoff": "ECUT",
    "cutoff energy": "ECUT",
    "boundary condition": "BC",
    "boundary conditions": "BC",
    "lattice vector": "LATVEC",
    "lattice vectors": "LATVEC",
    "lattice scale": "LATVEC_SCALE",
    "cell size": "CELL",
    "cell dimensions": "CELL",
    "k point grid": "KPOINT_GRID",
    "kpoint grid": "KPOINT_GRID",
    "k point shift": "KPOINT_SHIFT",
    "scf tolerance": "TOL_SCF",
    "scf max iterations": "MAXIT_SCF",
    "exchange correlation": "EXCHANGE_CORRELATION",
    "xc functional": "EXCHANGE_CORRELATION",
    "spin type": "SPIN_TYP",
    "electronic temperature": "ELEC_TEMP",
    "smearing": "SMEARING",
    "finite difference order": "FD_ORDER",
    "fd order": "FD_ORDER",
    "print forces": "PRINT_FORCES",
    "print atoms": "PRINT_ATOMS",
    "calculate stress": "CALC_STRESS",
}


def normalize_text(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(text.replace("_", " ").replace("-", " ").lower()))


def _clean_latex(text: str) -> str:
    text = text.replace("\\_", "_")
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\hyperlink\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\\\", " ")
    text = text.replace("$", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass(frozen=True)
class ReferenceHit:
    file: str
    line: int
    text: str
    score: int


class MSPARCReference:
    """Deterministic knowledge base built from the user's local M-SPARC checkout.

    The model is not expected to remember M-SPARC syntax. At startup this class parses
    doc/Manual.tex into a compact parameter schema and scans official tests for real
    .inpt/.ion examples and pseudopotential choices. Routine requests use this in-memory
    knowledge directly, avoiding repeated repository searches and extra LLM calls.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._files: list[Path] = []
        self._line_cache: dict[Path, list[str]] = {}
        self.inpt_keywords: dict[str, list[dict]] = {}
        self.ion_keywords: dict[str, list[dict]] = {}
        self.manual_specs: dict[str, dict] = {}
        self._pseudo_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._index_repository()

    def _iter_reference_files(self):
        # Do not eagerly scan src/. It is large and unnecessary for normal generation.
        # Manual + official tests are the authoritative fast path.
        preferred_roots = [self.root / "doc", self.root / "tests"]
        extras = [self.root / "README.md", self.root / "ChangeLog"]
        seen: set[Path] = set()
        for base in preferred_roots:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in REFERENCE_TEXT_EXTENSIONS and path not in seen:
                    seen.add(path)
                    yield path
        for path in extras:
            if path.exists() and path.is_file() and path not in seen:
                yield path

    def _read_lines(self, path: Path) -> list[str]:
        if path not in self._line_cache:
            try:
                self._line_cache[path] = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                self._line_cache[path] = []
        return self._line_cache[path]

    def _record_directive(self, table: dict[str, list[dict]], key: str, path: Path, line_number: int, value: str) -> None:
        canonical = key.upper()
        examples = table.setdefault(canonical, [])
        if len(examples) < 12:
            examples.append({
                "file": str(path.relative_to(self.root)),
                "line": line_number,
                "value": value,
            })

    def _parse_manual(self) -> None:
        manual = self.root / "doc" / "Manual.tex"
        if not manual.exists():
            return
        text = manual.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for frame_match in FRAME_PATTERN.finditer(text):
            frame = frame_match.group(0)
            label = LABEL_PATTERN.search(frame)
            if not label:
                continue
            keyword = label.group(1).upper()
            # Input labels are uppercase directives. Non-input frames are harmless because
            # validation also checks official test usage or a recognizable parameter block.
            blocks = {}
            for block_match in BLOCK_PATTERN.finditer(frame):
                blocks[_clean_latex(block_match.group(1))] = _clean_latex(block_match.group(2))
            if not blocks:
                continue
            start_line = text.count("\n", 0, frame_match.start()) + 1
            spec = {
                "keyword": keyword,
                "type": blocks.get("Type"),
                "default": blocks.get("Default"),
                "unit": blocks.get("Unit"),
                "example": blocks.get("Example"),
                "description": blocks.get("Description"),
                "remark": blocks.get("Remark"),
                "file": "doc/Manual.tex",
                "start_line": start_line,
            }
            self.manual_specs[keyword] = spec

    def _index_repository(self) -> None:
        self._parse_manual()
        self._files = list(self._iter_reference_files())
        for path in self._files:
            suffix = path.suffix.lower()
            if suffix not in {".inpt", ".ion"}:
                continue
            table = self.inpt_keywords if suffix == ".inpt" else self.ion_keywords
            current_atom: str | None = None
            for line_number, line in enumerate(self._read_lines(path), 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                match = DIRECTIVE_PATTERN.match(line)
                if match:
                    key, value = match.group(1), match.group(2)
                    self._record_directive(table, key, path, line_number, value)
                if suffix == ".ion":
                    atom_match = ATOM_TYPE_PATTERN.match(line)
                    if atom_match:
                        current_atom = atom_match.group(1)
                    pseudo_match = PSEUDO_PATTERN.match(line)
                    if pseudo_match and current_atom:
                        self._pseudo_counts[current_atom.capitalize()][Path(pseudo_match.group(1)).name] += 1

    @property
    def known_inpt_keywords(self) -> set[str]:
        # Manual.tex documents both .inpt and .ion directives. Anything observed only
        # in official .ion files must not be accepted as an .inpt keyword.
        ion_only = (set(self.ion_keywords) - set(self.inpt_keywords)) | ION_ONLY_KEYWORDS
        return set(self.inpt_keywords) | (set(self.manual_specs) - ion_only)

    @property
    def known_ion_keywords(self) -> set[str]:
        manual_ion = {"ATOM_TYPE", "PSEUDO_POT", "N_TYPE_ATOM", "COORD", "COORD_FRAC", "RELAX", "SPIN", "ATOMIC_MASS"}
        return set(self.ion_keywords) | (manual_ion & set(self.manual_specs)) | manual_ion

    def stats(self) -> dict:
        return {
            "repository": str(self.root),
            "indexed_files": len(self._files),
            "manual_parameters": len(self.manual_specs),
            "known_inpt_keywords": len(self.known_inpt_keywords),
            "known_ion_keywords": len(self.known_ion_keywords),
            "pseudopotential_elements_from_tests": len(self._pseudo_counts),
            "knowledge_mode": "local Manual.tex schema + official test examples",
        }

    def known_inpt_keyword(self, keyword: str) -> bool:
        return keyword.strip().upper() in self.known_inpt_keywords

    def known_ion_keyword(self, keyword: str) -> bool:
        return keyword.strip().upper() in self.known_ion_keywords

    def parameter_spec(self, keyword: str) -> dict:
        canonical = keyword.strip().upper()
        spec = dict(self.manual_specs.get(canonical, {}))
        if not spec:
            return {
                "keyword": canonical,
                "found": False,
                "examples": self.inpt_keywords.get(canonical, []),
            }
        spec["found"] = True
        spec["examples"] = self.inpt_keywords.get(canonical, [])[:6]
        return spec

    def default_value(self, keyword: str) -> str | None:
        default = self.manual_specs.get(keyword.strip().upper(), {}).get("default")
        if not default or default.strip().lower() in {"none", "n/a", "na"}:
            return None
        return default.strip()

    def keyword_examples(self, keyword: str, file_type: str = "inpt") -> dict:
        canonical = keyword.strip().upper()
        table = self.inpt_keywords if file_type.lower().lstrip(".") == "inpt" else self.ion_keywords
        return {
            "keyword": canonical,
            "known": self.known_inpt_keyword(canonical) if file_type.lower().lstrip(".") == "inpt" else self.known_ion_keyword(canonical),
            "examples": table.get(canonical, []),
            "manual": self.manual_specs.get(canonical),
        }

    def resolve_input_parameter(self, concept: str, max_candidates: int = 8) -> dict:
        normalized = normalize_text(concept)
        if normalized in COMMON_ALIASES:
            keyword = COMMON_ALIASES[normalized]
            return {
                "concept": concept,
                "resolved": keyword if self.known_inpt_keyword(keyword) else None,
                "candidates": [{"keyword": keyword, "score": 100, "manual": self.parameter_spec(keyword)}] if self.known_inpt_keyword(keyword) else [],
            }

        # Exact keyword typed by the user.
        exact = concept.strip().upper().replace(" ", "_")
        if self.known_inpt_keyword(exact):
            return {"concept": concept, "resolved": exact, "candidates": [{"keyword": exact, "score": 100, "manual": self.parameter_spec(exact)}]}

        terms = set(normalized.split())
        scores: dict[str, int] = {}
        for keyword in self.known_inpt_keywords:
            spec = self.manual_specs.get(keyword, {})
            searchable = normalize_text(" ".join([
                keyword,
                spec.get("description") or "",
                spec.get("remark") or "",
                spec.get("example") or "",
            ]))
            search_terms = set(searchable.split())
            overlap = len(terms & search_terms)
            if overlap:
                scores[keyword] = overlap * 10
                if normalized and normalized in searchable:
                    scores[keyword] += 25
        ranked = sorted(scores, key=lambda key: (-scores[key], key))[:max_candidates]
        resolved = ranked[0] if ranked and (len(ranked) == 1 or scores[ranked[0]] >= scores[ranked[1]] + 10) else None
        return {
            "concept": concept,
            "resolved": resolved,
            "candidates": [
                {"keyword": key, "score": scores[key], "manual": self.parameter_spec(key)} for key in ranked
            ],
        }

    def manual_parameter_help(self, keyword: str) -> dict:
        return self.parameter_spec(keyword)

    def preferred_pseudopotential(self, element: str) -> dict:
        symbol = element.strip().capitalize()
        psps = self.root / "psps"
        if not psps.exists():
            return {"element": symbol, "found": False, "reason": "M-SPARC psps/ directory not found", "candidates": []}

        candidates = [p for p in psps.glob("*.psp8") if f"_{symbol.lower()}_" in p.name.lower()]
        if not candidates:
            candidates = [p for p in psps.rglob("*.psp8") if f"_{symbol.lower()}_" in p.name.lower()]

        preferred_name = None
        counts = self._pseudo_counts.get(symbol)
        if counts:
            for name, _count in counts.most_common():
                match = next((p for p in candidates if p.name == name), None)
                if match:
                    preferred_name = match
                    break

        chosen = preferred_name or (candidates[0] if len(candidates) == 1 else None)
        return {
            "element": symbol,
            "found": chosen is not None,
            "chosen": str(chosen) if chosen else None,
            "source": "most frequent official test reference" if preferred_name else ("unique local psps match" if chosen else None),
            "candidates": [str(p) for p in sorted(candidates)[:20]],
        }

    def search(self, query: str, relative_dir: str = ".", max_matches: int = 40) -> list[dict]:
        query_normalized = normalize_text(query)
        terms = [term for term in query_normalized.split() if len(term) > 1]
        if not terms:
            return []
        base = (self.root / relative_dir).resolve()
        if base != self.root and self.root not in base.parents:
            raise ValueError("Reference path escapes the M-SPARC repository.")
        hits: list[ReferenceHit] = []
        for path in self._files:
            try:
                path.relative_to(base)
            except ValueError:
                continue
            relative = str(path.relative_to(self.root))
            file_priority = 3 if relative.startswith("doc") else 2
            for line_number, line in enumerate(self._read_lines(path), 1):
                normalized = normalize_text(line)
                matched = sum(1 for term in terms if term in normalized)
                if not matched:
                    continue
                score = matched * 2 + (5 if query_normalized in normalized else 0) + (3 if matched == len(terms) else 0) + file_priority
                hits.append(ReferenceHit(relative, line_number, line.strip(), score))
        hits.sort(key=lambda hit: (-hit.score, hit.file, hit.line))
        return [{"file": h.file, "line": h.line, "text": h.text, "score": h.score} for h in hits[:max_matches]]

    def discover_reference_cases(self, query: str = "", max_results: int = 40) -> list[dict]:
        tests_root = self.root / "tests"
        if not tests_root.exists():
            return []
        query_terms = set(normalize_text(query).split()) if query.strip() else set()
        results = []
        for inpt in sorted(tests_root.rglob("*.inpt")):
            stem = inpt.with_suffix("")
            ion = Path(str(stem) + ".ion")
            if not ion.exists():
                continue
            relative_stem = str(stem.relative_to(self.root))
            searchable = normalize_text(relative_stem)
            if query_terms and not all(term in searchable for term in query_terms):
                content = normalize_text(" ".join(self._read_lines(inpt)[:180] + self._read_lines(ion)[:220]))
                if not all(term in content for term in query_terms):
                    continue
            results.append({
                "case_stem": relative_stem,
                "inpt": str(inpt.relative_to(self.root)),
                "ion": str(ion.relative_to(self.root)),
            })
            if len(results) >= max_results:
                break
        return results
