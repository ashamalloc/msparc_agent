from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    workspace_root: Path
    msparc_root: Path
    api_key: str = field(repr=False)
    model: str = "gemini-3.8-flash"
    thinking_level: str = "low"
    max_agent_steps: int = 12

    @property
    def msparc_src(self) -> Path:
        return self.msparc_root / "src"

    def validate(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Gemini API key is required.")
        if not self.workspace_root.exists():
            raise FileNotFoundError(f"Workspace does not exist: {self.workspace_root}")
        for required in [self.msparc_root, self.msparc_src, self.msparc_root / "doc", self.msparc_root / "tests", self.msparc_root / "psps"]:
            if not required.exists():
                raise FileNotFoundError(f"Required M-SPARC path does not exist: {required}")
        if self.thinking_level not in {"low", "medium", "high"}:
            raise ValueError("thinking_level must be low, medium, or high")
        if self.max_agent_steps < 1:
            raise ValueError("max_agent_steps must be at least 1")
