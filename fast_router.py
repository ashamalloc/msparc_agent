from __future__ import annotations

import re
from pathlib import Path

from output_reader import QUANTITY_LABELS, normalize_quantity
from tools import Toolbox


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


class FastRouter:
    """Handle common requests locally with zero Gemini calls."""

    def __init__(self, tools: Toolbox):
        self.tools = tools

    def _cases(self) -> list[dict]:
        try:
            return self.tools.workspace.discover_cases()
        except Exception:
            return []

    def _mentioned_cases(self, message: str, cases: list[dict]) -> list[str]:
        lower = message.lower()
        matched = []
        # Longest names first so a nested/similar name wins cleanly.
        for case in sorted(cases, key=lambda item: len(item["case_stem"]), reverse=True):
            stem = case["case_stem"]
            name = Path(stem).name
            def present(token: str) -> bool:
                pattern = rf"(?<![A-Za-z0-9_]){re.escape(token.lower())}(?![A-Za-z0-9_])"
                return re.search(pattern, lower) is not None
            if present(stem) or present(name):
                matched.append(stem)
        # preserve uniqueness
        return list(dict.fromkeys(matched))

    def _format_query(self, result: dict) -> str:
        if not result.get("success"):
            return result.get("error", "The requested value was not found.")
        value = result.get("value")
        numbers = result.get("numbers") or []
        shown = value if value is not None else numbers
        return (
            f"{result['quantity'].title()} for `{result['case']}`: **{shown}**\n\n"
            f"Source: `{result['source_file']}`, line {result['source_line']}  \n"
            f"`{result['source_text']}`"
        )

    def _extract_sweep_values(self, message: str, mentioned_case: str) -> list[str]:
        cleaned = message.replace(mentioned_case, " ").replace(Path(mentioned_case).name, " ")
        # Prefer text after 'spacing(s)' or 'values' to avoid numbers embedded elsewhere.
        match = re.search(r"(?:mesh\s+spacings?|values?)\s*(?:of|:|=|are|for)?\s*([^.;\n]+)", cleaned, re.IGNORECASE)
        segment = match.group(1) if match else cleaned
        segment = re.sub(r"tolerance\s*(?:of|:|=)?\s*[-+0-9.eE]+", " ", segment, flags=re.IGNORECASE)
        values = NUMBER_RE.findall(segment)
        # Filter obvious integers from generic prose only when we have enough decimal values.
        decimal_like = [v for v in values if "." in v or "e" in v.lower()]
        return decimal_like if len(decimal_like) >= 2 else values

    def route(self, message: str) -> tuple[bool, str]:
        text = message.strip()
        lower = text.lower()
        if not text:
            return True, "Please enter a request."
        if lower in {"hi", "hii", "hello", "hey", "good morning", "good evening"}:
            return True, "Hello. I can create/validate M-SPARC cases, run them in local MATLAB, and read the generated outputs."

        cases = self._cases()
        mentioned = self._mentioned_cases(text, cases)

        if ("list" in lower or "show" in lower or "find" in lower) and "case" in lower and not mentioned:
            valid = [case for case in cases if case["valid_pair"]]
            if not cases:
                return True, "No `.inpt` cases were found in the configured simulation workspace."
            lines = [f"Found **{len(cases)}** case(s), **{len(valid)}** with matching `.inpt` + `.ion` files:"]
            lines.extend(f"- `{case['case_stem']}`" + ("" if case["valid_pair"] else " (missing .ion)") for case in cases)
            return True, "\n".join(lines)

        # Parameter documentation/default lookup can be answered from the local manual.
        if any(word in lower for word in ["syntax", "default", "manual", "parameter", "keyword"]) and not mentioned:
            # Remove generic question words and try the full phrase first; aliases handle common concepts.
            concept = lower
            for token in ["what is", "show me", "tell me", "the", "msparc", "manual", "syntax", "default", "value", "parameter", "keyword", "for"]:
                concept = concept.replace(token, " ")
            concept = " ".join(concept.split())
            if concept:
                info = self.tools.get_msparc_parameter_help(concept)
                manual = info.get("manual")
                if manual and manual.get("found"):
                    return True, (
                        f"`{info['resolved']}`\n\n"
                        f"- Type: {manual.get('type') or 'not stated'}\n"
                        f"- Default: {manual.get('default') or 'not stated'}\n"
                        f"- Unit: {manual.get('unit') or 'not stated'}\n"
                        f"- Example: {manual.get('example') or 'not stated'}\n"
                        f"- {manual.get('description') or ''}"
                    )

        if len(mentioned) == 1:
            case = mentioned[0]

            if "validate" in lower or "check input" in lower or "check case" in lower:
                result = self.tools.validate_case(case)
                if result["valid_for_run"]:
                    return True, f"`{case}` passed local M-SPARC validation and its pseudopotential paths exist."
                return True, f"`{case}` is not ready to run:\n- " + "\n- ".join(result["errors"])

            if any(word in lower for word in ["summary", "summarize", "summarise"]) and "output" in lower:
                result = self.tools.summarize_output(case)
                if not result.get("results"):
                    return True, f"No common summary quantities were found in the outputs for `{case}`."
                lines = [f"Output summary for `{case}`:"]
                for item in result["results"]:
                    lines.append(f"- {item['quantity']}: {item.get('value')} (`{item['source_file']}`, line {item['source_line']})")
                return True, "\n".join(lines)

            if "output" in lower and any(word in lower for word in ["file", "files", "generated", "show", "list"]):
                result = self.tools.inspect_outputs(case)
                files = result["files"]
                return True, (f"Output files for `{case}`:\n" + "\n".join(f"- `{path}`" for path in files)) if files else f"No recognized output files exist yet for `{case}`."

            # Direct output quantities: no LLM required.
            for alias in sorted(QUANTITY_LABELS, key=len, reverse=True):
                if alias in normalize_quantity(lower):
                    result = self.tools.query_output(case, alias)
                    return True, self._format_query(result)

            # Full convergence/sweep macro. It creates/runs/extracts/plots locally in one call.
            if "mesh" in lower and ("sweep" in lower or "convergence" in lower):
                values = self._extract_sweep_values(text, case)
                if len(values) >= 2:
                    should_run = any(word in lower for word in ["run", "execute", "perform", "convergence"])
                    if should_run:
                        tolerance = -1.0
                        tol_match = re.search(r"tolerance\s*(?:of|:|=)?\s*([-+0-9.eE]+)", lower)
                        if tol_match:
                            try:
                                tolerance = float(tol_match.group(1))
                            except ValueError:
                                pass
                        result = self.tools.run_parameter_convergence_study(case, "mesh spacing", values, tolerance=tolerance)
                        if not result.get("success"):
                            # Partial runs are still useful; return the concrete failure.
                            failed = [r for r in result.get("runs", []) if not r.get("success")]
                            if failed:
                                return True, f"The sweep was prepared, but {len(failed)} run(s) failed. First error: {failed[0].get('error') or failed[0].get('stderr') or 'unknown error'}"
                        lines = [f"Completed mesh study for `{case}`."]
                        if result.get("plot_path"):
                            lines.append(f"Plot: `{result['plot_path']}`")
                        if result.get("y"):
                            lines.append(f"Extracted values: {result['y']}")
                        if result.get("successive_absolute_differences"):
                            lines.append(f"Successive |Δ|: {result['successive_absolute_differences']}")
                        if result.get("convergence") is not None:
                            lines.append(f"Converged from x = {result['convergence'].get('converged_from_x')} using tolerance {tolerance}.")
                        elif tolerance < 0:
                            lines.append("No convergence tolerance was supplied, so no threshold was invented.")
                        return True, "\n\n".join(lines)
                    prepared = self.tools.prepare_parameter_sweep(case, "mesh spacing", values)
                    if prepared.get("success"):
                        return True, f"Prepared {len(prepared['cases'])} validated mesh-spacing cases under `sweeps/`."
                    return True, prepared.get("error", "Sweep preparation failed.")

            # Direct execution of an existing case.
            if any(word in lower for word in ["run", "execute"]) and not any(word in lower for word in ["create", "sweep", "convergence"]):
                result = self.tools.run_msparc(case)
                if result.get("success"):
                    outputs = result.get("output_files", [])
                    return True, f"M-SPARC completed `{case}` in local MATLAB. Generated/recognized outputs:\n" + "\n".join(f"- `{path}`" for path in outputs)
                return True, f"M-SPARC did not complete `{case}`: {result.get('error') or result.get('stderr') or 'unknown error'}"

        return False, ""
