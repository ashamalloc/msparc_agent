from __future__ import annotations

import json
from pathlib import Path

from analysis import convergence_report, save_xy_plot
from case_builder import GRID_CONTROL, LATTICE_SCALE_CONTROL, MSPARCCaseBuilder
from config import AgentConfig
from matlab_runner import MatlabRunner
from output_reader import MSPARCOutputReader
from reference import MSPARCReference
from workspace import Workspace, safe_value_token


class Toolbox:
    """High-level local operations exposed to Gemini.

    v3 intentionally exposes fewer, larger tools. Most work is deterministic local Python;
    Gemini is used for language understanding, not for writing M-SPARC syntax line-by-line.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.workspace = Workspace(config.workspace_root)
        self.reference_workspace = Workspace(config.msparc_root)
        self.reference = MSPARCReference(config.msparc_root)
        self.builder = MSPARCCaseBuilder(self.workspace, self.reference, config.msparc_root)
        self.outputs = MSPARCOutputReader(self.workspace)
        self.matlab = MatlabRunner(config.msparc_src)
        self.artifacts: list[str] = []

    def reference_status(self) -> dict:
        """Report the local M-SPARC manual/test knowledge loaded at startup."""
        return self.reference.stats()

    def discover_cases(self, relative_dir: str = ".") -> dict:
        """List M-SPARC cases in the simulation workspace.

        Args:
            relative_dir: Directory relative to the configured simulation workspace.
        """
        cases = self.workspace.discover_cases(relative_dir)
        return {"cases": cases, "valid_cases": [case["case_stem"] for case in cases if case["valid_pair"]]}

    def inspect_case(self, case_stem: str) -> dict:
        """Inspect one case's input files, parameters, atoms, pseudopotentials, and outputs.

        Args:
            case_stem: Relative case path without .inpt/.ion extension.
        """
        case = self.workspace.case_files(case_stem)
        if not case["valid_pair"]:
            return {"success": False, "error": f"{case_stem} is missing a same-stem .inpt/.ion pair.", "case": case}
        return {
            "success": True,
            "case": case,
            "parameters": self.workspace.read_inpt_parameters(case_stem),
            "ion_blocks": self.workspace.parse_ion_blocks(case_stem),
            "pseudopotentials": self.workspace.pseudopotential_references(case_stem),
            "outputs": self.workspace.output_files(case_stem),
        }

    def get_msparc_parameter_help(self, parameter_or_concept: str) -> dict:
        """Resolve a human phrase and return exact Manual.tex syntax, type, default, example, and remarks.

        Args:
            parameter_or_concept: Exact keyword or phrase such as mesh spacing or SCF tolerance.
        """
        resolved = self.reference.resolve_input_parameter(parameter_or_concept)
        keyword = resolved.get("resolved")
        if keyword:
            return {"resolved": keyword, "manual": self.reference.parameter_spec(keyword)}
        return resolved

    def create_case_from_spec(self, case_stem: str, spec_json: str, overwrite: bool = False) -> dict:
        """Create correct .inpt/.ion files from structured scientific data.

        Gemini must provide semantic JSON, never raw file text. The local builder validates every
        .inpt keyword against Manual.tex/tests, renders official syntax, resolves pseudopotentials
        from local official evidence, and refuses incomplete/ambiguous cases.

        JSON shape:
        {"inpt": {"LATVEC": [[...],[...],[...]], "LATVEC_SCALE": [...],
                  "MESH_SPACING": 0.2, "BC": ["P","P","P"], ...},
         "species": [{"element":"Fe", "coordinate_mode":"fractional",
                      "coordinates":[[0,0,0]], "pseudopotential":"auto",
                      "spin":[1.0], "relax":[[1,1,1]]}]}

        Args:
            case_stem: New relative case path without extension.
            spec_json: JSON object describing input parameters and atomic species.
            overwrite: Replace an existing case only when explicitly requested.
        """
        result = self.builder.create_case(case_stem, spec_json, overwrite)
        if result.get("success"):
            result["validation"] = self._validate_case(case_stem)
        return result

    def modify_case_parameters(self, case_stem: str, parameters_json: str) -> dict:
        """Modify existing single-line .inpt parameters using local Manual.tex validation.

        Conflicting grid controls are removed automatically (for example setting MESH_SPACING
        removes FD_GRID/ECUT). The change is transactional: if validation fails, the original
        .inpt file is restored.

        Args:
            case_stem: Existing case stem.
            parameters_json: JSON object such as {"MESH_SPACING":0.25,"TOL_SCF":1e-6}.
        """
        try:
            changes = json.loads(parameters_json)
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"parameters_json is invalid JSON: {exc}"}
        if not isinstance(changes, dict) or not changes:
            return {"success": False, "error": "parameters_json must be a non-empty JSON object."}

        stem = self.workspace.resolve(case_stem)
        inpt_path = Path(str(stem) + ".inpt")
        if not inpt_path.exists():
            return {"success": False, "error": f"Input file not found for {case_stem}."}
        original_text = inpt_path.read_text(encoding="utf-8", errors="replace")
        results = []
        removals = []
        try:
            for raw_key, value in changes.items():
                key = str(raw_key).upper().strip()
                if not self.reference.known_inpt_keyword(key):
                    raise ValueError(f"Unknown M-SPARC input keyword: {key}")
                if key == "LATVEC":
                    raise ValueError("LATVEC is multiline; create from a structured spec or a validated template instead of line replacement.")
                if key in GRID_CONTROL:
                    for conflict in sorted(GRID_CONTROL - {key}):
                        removals.append(self.workspace.remove_inpt_parameter(case_stem, conflict))
                if key == "CELL":
                    removals.append(self.workspace.remove_inpt_parameter(case_stem, "LATVEC_SCALE"))
                elif key == "LATVEC_SCALE":
                    removals.append(self.workspace.remove_inpt_parameter(case_stem, "CELL"))
                value_text = " ".join(str(x) for x in value) if isinstance(value, list) else str(value)
                results.append(self.workspace.set_inpt_parameter(case_stem, key, value_text, allow_append=True))

            validation = self._validate_case(case_stem)
            if not validation["valid_for_run"]:
                inpt_path.write_text(original_text, encoding="utf-8")
                return {
                    "success": False,
                    "error": "Requested changes would leave the case invalid; original .inpt was restored.",
                    "attempted_changes": results,
                    "attempted_validation": validation,
                }
            return {"success": True, "changes": results, "removed_conflicts": [r for r in removals if r.get("count")], "validation": validation}
        except Exception as exc:
            inpt_path.write_text(original_text, encoding="utf-8")
            return {"success": False, "error": str(exc), "note": "Original .inpt was restored."}

    def duplicate_case(self, source_stem: str, destination_stem: str) -> dict:
        """Copy an existing valid case and repair relative pseudopotential paths.

        Args:
            source_stem: Existing case stem.
            destination_stem: New case stem.
        """
        result = self.workspace.duplicate_case(source_stem, destination_stem)
        result["validation"] = self._validate_case(destination_stem)
        return result

    def find_reference_cases(self, query: str = "", max_results: int = 20) -> dict:
        """Find official M-SPARC test cases as trustworthy templates.

        Args:
            query: Material/method words such as Fe spin or BaTiO3.
            max_results: Maximum candidates.
        """
        return {"cases": self.reference.discover_reference_cases(query, max_results)}

    def copy_reference_case(self, reference_stem: str, destination_stem: str) -> dict:
        """Copy an official test case into the writable workspace and repair pseudopotential paths.

        Args:
            reference_stem: Case stem relative to the M-SPARC repository, normally tests/.../case.
            destination_stem: New workspace case stem.
        """
        source = self.reference_workspace.resolve(reference_stem)
        result = self.workspace.copy_external_case(source, destination_stem)
        result["source_reference"] = reference_stem
        result["validation"] = self._validate_case(destination_stem)
        return result

    def _validate_case(self, case_stem: str) -> dict:
        case = self.workspace.case_files(case_stem)
        errors: list[str] = []
        warnings: list[str] = []
        if not case["valid_pair"]:
            return {"case": case_stem, "valid_for_run": False, "errors": ["Missing same-stem .inpt/.ion pair."], "warnings": []}

        inpt = self.workspace.read_directives(case["inpt"])
        unknown_inpt = [item for item in inpt if not self.reference.known_inpt_keyword(item["parameter"])]
        if unknown_inpt:
            errors.append("Unknown .inpt directive(s) not found in local Manual.tex/tests: " + ", ".join(item["parameter"] for item in unknown_inpt))

        parameter_names = {item["parameter"].upper() for item in inpt}
        grid = GRID_CONTROL & parameter_names
        if len(grid) != 1:
            errors.append("Exactly one of MESH_SPACING, FD_GRID, or ECUT must be specified.")
        scale_controls = LATTICE_SCALE_CONTROL & parameter_names
        if len(scale_controls) > 1:
            errors.append("CELL and LATVEC_SCALE cannot be specified simultaneously.")
        if not ({"CELL", "LATVEC", "LATVEC_SCALE"} & parameter_names):
            errors.append("No simulation-cell definition was found (CELL/LATVEC/LATVEC_SCALE).")
        if "BC" not in parameter_names:
            errors.append("BC is missing and the M-SPARC manual gives it no default.")

        blocks = self.workspace.parse_ion_blocks(case_stem)
        if not blocks:
            errors.append(".ion file contains no ATOM_TYPE blocks.")
        for index, block in enumerate(blocks, 1):
            atom = block.get("atom_type") or f"block {index}"
            if not block.get("atom_type"):
                errors.append(f"Ion block {index} is missing ATOM_TYPE.")
            if not block.get("pseudopotential"):
                errors.append(f"{atom}: missing PSEUDO_POT.")
            n_atoms = block.get("n_type_atom")
            if not isinstance(n_atoms, int) or n_atoms < 1:
                errors.append(f"{atom}: N_TYPE_ATOM must be a positive integer.")
                continue
            coord = block.get("coord")
            frac = block.get("coord_frac")
            if bool(coord) == bool(frac):
                errors.append(f"{atom}: specify exactly one of COORD or COORD_FRAC.")
            coordinate_rows = coord or frac or []
            if len(coordinate_rows) != n_atoms:
                errors.append(f"{atom}: N_TYPE_ATOM={n_atoms} but found {len(coordinate_rows)} coordinate row(s).")
            if block.get("spin") is not None and len(block["spin"]) != n_atoms:
                errors.append(f"{atom}: SPIN count does not match N_TYPE_ATOM.")
            if block.get("relax") is not None and len(block["relax"]) != n_atoms:
                errors.append(f"{atom}: RELAX count does not match N_TYPE_ATOM.")

        pseudopotentials = self.workspace.pseudopotential_references(case_stem)
        missing = [item for item in pseudopotentials if not item["exists"]]
        if missing:
            errors.append("One or more PSEUDO_POT paths do not exist from the case directory.")
        if not pseudopotentials:
            errors.append("No .psp8 pseudopotential path was found in the .ion file.")

        return {
            "case": case_stem,
            "valid_for_run": not errors,
            "errors": errors,
            "warnings": warnings,
            "unknown_inpt_directives": unknown_inpt,
            "pseudopotentials": pseudopotentials,
            "ion_blocks": blocks,
        }

    def validate_case(self, case_stem: str) -> dict:
        """Validate syntax structure and pseudopotential paths before MATLAB execution.

        Args:
            case_stem: Relative case stem.
        """
        return self._validate_case(case_stem)

    def prepare_parameter_sweep(self, source_stem: str, parameter: str, values: list[str], sweep_dir: str = "sweeps") -> dict:
        """Create validated independent sweep cases locally, without LLM repetition.

        Args:
            source_stem: Existing valid base case.
            parameter: Exact or common parameter phrase.
            values: Values for independent runs.
            sweep_dir: Destination directory.
        """
        resolved = self.reference.resolve_input_parameter(parameter)
        canonical = resolved.get("resolved")
        if not canonical:
            return {"success": False, "error": f"Could not uniquely resolve parameter {parameter!r} from the local manual.", "reference": resolved}
        if canonical == "LATVEC":
            return {"success": False, "error": "Multiline LATVEC sweeps require a dedicated structured workflow."}
        source_validation = self._validate_case(source_stem)
        if not source_validation["valid_for_run"]:
            return {"success": False, "error": "Source case failed validation.", "validation": source_validation}

        source_name = Path(source_stem).name
        created = []
        for value in values:
            destination = f"{sweep_dir}/{source_name}_{canonical.lower()}_{safe_value_token(value)}"
            copy_result = self.workspace.duplicate_case(source_stem, destination, reuse_existing=True)
            if canonical in GRID_CONTROL:
                for conflict in GRID_CONTROL - {canonical}:
                    self.workspace.remove_inpt_parameter(destination, conflict)
            if canonical == "CELL":
                self.workspace.remove_inpt_parameter(destination, "LATVEC_SCALE")
            elif canonical == "LATVEC_SCALE":
                self.workspace.remove_inpt_parameter(destination, "CELL")
            change = self.workspace.set_inpt_parameter(destination, canonical, str(value), allow_append=True)
            validation = self._validate_case(destination)
            created.append({"case_stem": destination, "value": value, "copy": copy_result, "change": change, "validation": validation})
        return {"success": all(item["validation"]["valid_for_run"] for item in created), "parameter": canonical, "cases": created}

    def run_msparc(self, case_stem: str, parallel_switch: bool = False) -> dict:
        """Run one validated M-SPARC case in local MATLAB and report generated outputs.

        Args:
            case_stem: Relative case stem.
            parallel_switch: Enable M-SPARC parallel k-point/spin switch when appropriate.
        """
        validation = self._validate_case(case_stem)
        if not validation["valid_for_run"]:
            return {"success": False, "error": "Case validation failed; MATLAB was not started.", "validation": validation}
        before = set(self.workspace.output_files(case_stem))
        result = self.matlab.run(self.workspace.resolve(case_stem), parallel_switch)
        after = set(self.workspace.output_files(case_stem))
        result["validation"] = validation
        result["output_files"] = sorted(after)
        result["new_output_files"] = sorted(after - before)
        result["newest_output"] = self.workspace.newest_output_for_case(case_stem)
        return result

    def run_msparc_batch(self, case_stems: list[str], parallel_switch: bool = False) -> dict:
        """Run several M-SPARC cases sequentially in one persistent MATLAB session.

        Args:
            case_stems: Relative case stems.
            parallel_switch: M-SPARC parallel switch.
        """
        results = [self.run_msparc(case, parallel_switch) for case in case_stems]
        return {"success": all(item.get("success") for item in results), "results": results}

    def inspect_outputs(self, case_stem: str) -> dict:
        """List recognized output files for a case.

        Args:
            case_stem: Relative case stem.
        """
        return {"case": case_stem, "files": self.workspace.output_files(case_stem), "newest_output": self.workspace.newest_output_for_case(case_stem)}

    def query_output(self, case_stem: str, quantity: str) -> dict:
        """Read a requested quantity directly from local M-SPARC output files.

        Args:
            case_stem: Relative case stem.
            quantity: Quantity such as total energy, Fermi level, pressure, maximum force, magnetization, SCF iterations, or walltime.
        """
        return self.outputs.query(case_stem, quantity)

    def query_output_batch(self, case_stems: list[str], quantity: str) -> dict:
        """Read the same output quantity from multiple cases locally.

        Args:
            case_stems: Relative case stems.
            quantity: Output quantity.
        """
        return self.outputs.query_many(case_stems, quantity)

    def summarize_output(self, case_stem: str) -> dict:
        """Read a compact set of common final quantities from one M-SPARC output locally.

        Args:
            case_stem: Relative case stem.
        """
        quantities = ["total energy", "fermi level", "maximum force", "pressure", "magnetization", "scf iterations", "walltime"]
        results = [self.outputs.query(case_stem, quantity) for quantity in quantities]
        return {"case": case_stem, "results": [item for item in results if item.get("success")]}

    def run_parameter_convergence_study(
        self,
        source_stem: str,
        parameter: str,
        values: list[str],
        quantity: str = "total energy",
        tolerance: float = -1.0,
        sweep_dir: str = "sweeps",
        plot_file: str = "outputs/convergence.png",
    ) -> dict:
        """Create, run, extract, and plot a complete convergence study in one local call.

        This macro is deliberately large to avoid many Gemini round-trips.

        Args:
            source_stem: Valid base case.
            parameter: Parameter name/concept, e.g. mesh spacing.
            values: Sweep values.
            quantity: Output quantity to extract after successful runs.
            tolerance: Optional convergence tolerance; use a negative value when not supplied.
            sweep_dir: Directory for generated cases.
            plot_file: PNG output path inside workspace.
        """
        prepared = self.prepare_parameter_sweep(source_stem, parameter, values, sweep_dir)
        if not prepared.get("success"):
            return {"success": False, "error": "Sweep preparation failed.", "prepared": prepared}
        cases = [item["case_stem"] for item in prepared["cases"]]
        runs = self.run_msparc_batch(cases)["results"]

        x = []
        y = []
        extracted = []
        for raw_value, case, run in zip(values, cases, runs):
            if not run.get("success"):
                extracted.append({"case": case, "success": False, "run_error": run.get("error")})
                continue
            query = self.outputs.query(case, quantity)
            extracted.append(query)
            if query.get("success") and query.get("value") is not None:
                try:
                    x.append(float(raw_value))
                    y.append(float(query["value"]))
                except (TypeError, ValueError):
                    pass

        plot_path = None
        differences = []
        convergence = None
        if len(x) >= 2 and len(x) == len(y):
            output = self.workspace.resolve(plot_file)
            save_xy_plot(x, y, output, parameter, quantity, f"{quantity.title()} vs {parameter}")
            plot_path = str(output.relative_to(self.workspace.root))
            self.artifacts.append(plot_path)
            differences = [abs(y[i] - y[i - 1]) for i in range(1, len(y))]
            if tolerance >= 0:
                convergence = convergence_report(x, y, tolerance)

        return {
            "success": all(run.get("success") for run in runs) and len(x) == len(values),
            "parameter": prepared.get("parameter"),
            "values": values,
            "case_stems": cases,
            "runs": runs,
            "extracted": extracted,
            "x": x,
            "y": y,
            "successive_absolute_differences": differences,
            "convergence": convergence,
            "formal_convergence_note": None if tolerance >= 0 else "No tolerance was supplied, so the agent reports differences but does not invent a convergence threshold.",
            "plot_path": plot_path,
        }

    def plot_xy(self, x: list[float], y: list[float], x_label: str, y_label: str, title: str, output_file: str = "outputs/plot.png") -> dict:
        """Create a PNG line plot from verified numeric values.

        Args:
            x: X values.
            y: Y values.
            x_label: X-axis label.
            y_label: Y-axis label.
            title: Plot title.
            output_file: Relative PNG path in workspace.
        """
        output = self.workspace.resolve(output_file)
        if output.suffix.lower() != ".png":
            return {"success": False, "error": "output_file must end in .png"}
        save_xy_plot(x, y, output, x_label, y_label, title)
        relative = str(output.relative_to(self.workspace.root))
        self.artifacts.append(relative)
        return {"success": True, "plot_path": relative}
