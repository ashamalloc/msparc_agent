# Changelog

## 3.0.0

- Added deterministic `Manual.tex` knowledge parsing for parameter syntax, types, defaults, examples, descriptions, and remarks.
- Added structured `.inpt` / `.ion` generator. Gemini supplies semantic JSON only; local Python renders M-SPARC syntax.
- Added evidence-based pseudopotential selection from official M-SPARC tests and local `psps/` files.
- Added stronger case validation for grid controls, lattice definition, atom counts, coordinates, spin/relax counts, and pseudopotential paths.
- MATLAB now changes into the case directory before calling `msparc`, so relative pseudopotential paths resolve from the intended location.
- Added local output reader for total energy, Fermi level, forces, pressure, stress, magnetization, SCF iterations, and walltime.
- Added zero-LLM fast paths for listing/validating/running cases, reading common output quantities, and standard mesh-convergence studies.
- Replaced many low-level tools with high-level macro tools to reduce Gemini API calls and latency.
- Removed execution-step display from the Streamlit UI.
- Preserved safe handling of Gemini rate/quota failures so a later prompt cannot resume an unfinished older request.
