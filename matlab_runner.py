from __future__ import annotations

import io
from pathlib import Path


class MatlabRunner:
    def __init__(self, msparc_src: Path):
        self.msparc_src = msparc_src.resolve()
        self.engine = None

    def start(self) -> None:
        if self.engine is not None:
            return
        try:
            import matlab.engine
        except ImportError as exc:
            raise RuntimeError(
                "MATLAB Engine API for Python is not installed in this Python environment. "
                "Install the matlabengine package version matching the installed MATLAB release."
            ) from exc
        self.engine = matlab.engine.start_matlab()
        self.engine.addpath(str(self.msparc_src), nargout=0)

    def run(self, absolute_case_stem: Path, parallel_switch: bool = False) -> dict:
        """Run M-SPARC from the case directory.

        Running from the case directory matters because M-SPARC input files commonly use
        relative pseudopotential paths. The previous implementation called msparc with an
        absolute case path while leaving MATLAB's working directory unchanged, which could
        make otherwise-correct PSEUDO_POT paths resolve from the wrong directory.
        """
        self.start()
        stem = absolute_case_stem.resolve()
        case_dir = stem.parent
        case_name = stem.name
        stdout = io.StringIO()
        stderr = io.StringIO()
        previous_dir = self.engine.pwd()

        try:
            self.engine.cd(str(case_dir), nargout=0)
            if parallel_switch:
                self.engine.msparc(case_name, 1.0, nargout=0, stdout=stdout, stderr=stderr)
            else:
                self.engine.msparc(case_name, nargout=0, stdout=stdout, stderr=stderr)
            return {
                "success": True,
                "case": str(stem),
                "matlab_working_directory": str(case_dir),
                "stdout": stdout.getvalue()[-8000:],
                "stderr": stderr.getvalue()[-8000:],
            }
        except Exception as exc:
            return {
                "success": False,
                "case": str(stem),
                "matlab_working_directory": str(case_dir),
                "error": str(exc),
                "stdout": stdout.getvalue()[-8000:],
                "stderr": stderr.getvalue()[-8000:],
            }
        finally:
            try:
                self.engine.cd(previous_dir, nargout=0)
            except Exception:
                pass

    def close(self) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None
