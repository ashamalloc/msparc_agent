from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from google import genai

from config import AgentConfig
from fast_router import FastRouter
from tools import Toolbox


def _json_schema_for_type(annotation) -> dict:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if origin is list and args:
        return {"type": "array", "items": _json_schema_for_type(args[0])}
    return {"type": "string"}


def _tool_declaration(function) -> dict:
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    properties = {}
    required = []
    for name, parameter in signature.parameters.items():
        annotation = hints.get(name, str)
        properties[name] = _json_schema_for_type(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    declaration = {
        "type": "function",
        "name": function.__name__,
        "description": inspect.getdoc(function) or function.__name__,
        "parameters": {"type": "object", "properties": properties},
    }
    if required:
        declaration["parameters"]["required"] = required
    return declaration


def _looks_like_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).lower()
    return status == 429 or any(token in text for token in [
        "429", "resource_exhausted", "rate limit", "rate_limit_exceeded",
        "quota_exceeded", "too many requests", "request limit exceeded",
    ])


def _safe_json(result) -> str:
    text = json.dumps(result, default=str, ensure_ascii=False)
    return text if len(text) <= 30000 else text[:30000] + "\n... [local result truncated]"


class MSPARCAgent:
    """Hybrid agent: local deterministic fast paths first, Gemini only when language reasoning is needed."""

    def __init__(self, config: AgentConfig):
        config.validate()
        self.config = config
        self.tools = Toolbox(config)
        self.router = FastRouter(self.tools)
        self.system_prompt = (Path(__file__).parent / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
        self.client = genai.Client(api_key=config.api_key)

        # High-level tools only. Large local macro operations replace many model/tool/model cycles.
        self.all_tool_functions = [
            self.tools.reference_status,
            self.tools.discover_cases,
            self.tools.inspect_case,
            self.tools.get_msparc_parameter_help,
            self.tools.create_case_from_spec,
            self.tools.modify_case_parameters,
            self.tools.duplicate_case,
            self.tools.find_reference_cases,
            self.tools.copy_reference_case,
            self.tools.validate_case,
            self.tools.prepare_parameter_sweep,
            self.tools.run_msparc,
            self.tools.run_msparc_batch,
            self.tools.inspect_outputs,
            self.tools.query_output,
            self.tools.query_output_batch,
            self.tools.summarize_output,
            self.tools.run_parameter_convergence_study,
            self.tools.plot_xy,
        ]
        self.tool_map = {fn.__name__: fn for fn in self.all_tool_functions}
        self.declarations = {fn.__name__: _tool_declaration(fn) for fn in self.all_tool_functions}

        self.previous_interaction_id: str | None = None
        self.last_artifacts: list[str] = []
        self.last_error_kind: str | None = None
        self.turn_number = 0
        self.local_context: list[str] = []

    def reset_conversation(self) -> None:
        self.previous_interaction_id = None
        self.last_artifacts = []
        self.last_error_kind = None
        self.local_context = []

    def _thinking_level_for(self, message: str) -> str:
        if self.config.thinking_level != "low":
            return self.config.thinking_level
        lower = message.lower()
        # Keep routine actions fast. Escalate only genuinely compound generation/reasoning tasks.
        action_words = ["create", "generate", "modify", "compare", "analyze", "analyse", "then", "after", "and"]
        count = sum(lower.count(word) for word in action_words)
        if len(message) > 700 or ("create" in lower and count >= 4):
            return "medium"
        return "low"

    def _select_tool_names(self, message: str) -> list[str]:
        lower = message.lower()
        selected = {"discover_cases", "inspect_case", "get_msparc_parameter_help", "query_output"}
        if any(word in lower for word in ["create", "generate", ".inpt", ".ion", "atom", "coordinate", "lattice", "cell"]):
            selected |= {"create_case_from_spec", "find_reference_cases", "copy_reference_case", "validate_case", "duplicate_case"}
        if any(word in lower for word in ["change", "modify", "set", "parameter"]):
            selected |= {"modify_case_parameters", "validate_case", "duplicate_case"}
        if any(word in lower for word in ["run", "execute", "matlab", "simulate", "simulation"]):
            selected |= {"run_msparc", "run_msparc_batch", "validate_case", "inspect_outputs", "query_output_batch", "summarize_output"}
        if any(word in lower for word in ["compare", "summary", "summarize", "summarise"]):
            selected |= {"query_output_batch", "summarize_output", "inspect_outputs"}
        if any(word in lower for word in ["sweep", "convergence", "mesh"]):
            selected |= {"prepare_parameter_sweep", "run_parameter_convergence_study", "plot_xy", "query_output_batch"}
        if any(word in lower for word in ["plot", "graph", "visual"]):
            selected |= {"plot_xy", "query_output_batch"}
        if any(word in lower for word in ["reference", "template", "example", "manual", "default", "syntax"]):
            selected |= {"reference_status", "find_reference_cases", "copy_reference_case"}
        return [name for name in self.tool_map if name in selected]

    def _create_interaction(self, input_data, previous_interaction_id: str | None, thinking_level: str, tool_names: list[str]):
        args = {
            "model": self.config.model,
            "input": input_data,
            "system_instruction": self.system_prompt,
            "tools": [self.declarations[name] for name in tool_names],
            "generation_config": {"thinking_level": thinking_level},
        }
        if previous_interaction_id:
            args["previous_interaction_id"] = previous_interaction_id
        return self.client.interactions.create(**args)

    def _rate_limit_message(self) -> str:
        return (
            "Gemini's request/quota limit was reached before this task finished. "
            "The unfinished interaction was discarded, so your next prompt will not resume the old request. "
            "Most routine M-SPARC actions now run locally without Gemini, which reduces how often this should occur."
        )

    def chat(self, user_message: str) -> str:
        message = user_message.strip()
        if not message:
            return "Please enter a request."

        self.turn_number += 1
        artifact_start = len(self.tools.artifacts)
        self.last_artifacts = []
        self.last_error_kind = None

        # Zero-LLM path for common operations such as run case, read energy, list cases,
        # manual defaults, and standard mesh convergence studies.
        handled, local_answer = self.router.route(message)
        if handled:
            self.last_artifacts = self.tools.artifacts[artifact_start:]
            compact = f"User: {message}\nLocal result: {local_answer}"
            self.local_context = (self.local_context + [compact])[-2:]
            return local_answer

        committed_previous_id = self.previous_interaction_id
        working_previous_id = committed_previous_id
        thinking_level = self._thinking_level_for(message)
        tool_names = self._select_tool_names(message)
        recent_local = "\n\n".join(self.local_context[-2:])
        context_note = f"\n\nRECENT LOCAL CONTEXT (already completed, do not repeat):\n{recent_local}" if recent_local else ""
        turn_input = (
            f"CURRENT REQUEST:\n{message}"
            f"{context_note}\n\n"
            "Complete every explicit part of the current request. Prefer one high-level local tool over many small calls. "
            "Never emit raw .inpt/.ion syntax from memory; use create_case_from_spec or a validated template."
        )

        try:
            interaction = self._create_interaction(turn_input, working_previous_id, thinking_level, tool_names)
            for _round in range(self.config.max_agent_steps):
                calls = [step for step in (interaction.steps or []) if step.type == "function_call"]
                if not calls:
                    self.previous_interaction_id = interaction.id
                    self.last_artifacts = self.tools.artifacts[artifact_start:]
                    answer = (interaction.output_text or "").strip()
                    self.local_context = []
                    return answer or "Gemini completed the request but returned no text response."

                function_results = []
                for call in calls:
                    fn = self.tool_map.get(call.name)
                    arguments = call.arguments or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if fn is None:
                        result = {"success": False, "error": f"Unknown tool: {call.name}"}
                    else:
                        try:
                            result = fn(**arguments)
                        except Exception as exc:
                            result = {"success": False, "error": str(exc), "tool": call.name}
                    function_results.append({
                        "type": "function_result",
                        "name": call.name,
                        "call_id": call.id,
                        "result": [{"type": "text", "text": _safe_json(result)}],
                    })
                working_previous_id = interaction.id
                interaction = self._create_interaction(function_results, working_previous_id, thinking_level, tool_names)

            final_calls = [step for step in (interaction.steps or []) if step.type == "function_call"]
            if not final_calls:
                self.previous_interaction_id = interaction.id
                self.last_artifacts = self.tools.artifacts[artifact_start:]
                self.local_context = []
                return (interaction.output_text or "").strip() or "Task completed."

            self.previous_interaction_id = committed_previous_id
            self.last_error_kind = "max_steps"
            self.last_artifacts = self.tools.artifacts[artifact_start:]
            return "The task did not finish within the local agent-round limit. Completed local file/MATLAB actions remain on disk, but the unfinished Gemini state was not saved."

        except Exception as exc:
            self.previous_interaction_id = committed_previous_id
            self.last_artifacts = self.tools.artifacts[artifact_start:]
            if _looks_like_rate_limit(exc):
                self.previous_interaction_id = None
                self.local_context = []
                self.last_error_kind = "rate_limit"
                return self._rate_limit_message()
            self.last_error_kind = "api_error"
            return f"Gemini/API error: {exc}. The failed turn was not committed."
