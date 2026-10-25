# M-SPARC Natural-Language Agent — Gemini v3

This build is designed around the actual project objective: natural-language control of local M-SPARC/MATLAB simulations, followed by local output inspection and analysis.

## What v3 changes

### 1. Correct `.inpt` and `.ion` generation

The language model **does not write raw M-SPARC files**. Instead:

```text
user description
    ↓
Gemini extracts structured scientific data (JSON)
    ↓
local Manual.tex knowledge validates every input keyword
    ↓
local deterministic writer renders .inpt/.ion syntax
    ↓
local validator checks atoms, coordinates, grid settings and pseudopotentials
```

If essential data is missing or ambiguous, the agent asks rather than inventing it.

### 2. Local manual knowledge instead of repeated GitHub/repository searching

At startup the agent parses the configured local M-SPARC repository:

- `doc/Manual.tex` → keyword, type, default, unit, example, description, remark
- `tests/**/*.inpt` → official input examples
- `tests/**/*.ion` → official ion syntax and commonly used pseudopotentials
- `psps/` → actual local pseudopotential files

This is not model fine-tuning. It is safer and faster: the program builds an exact local schema from the same M-SPARC version that will be run.

### 3. Local MATLAB execution

`run_msparc` validates a case and launches the user's installed MATLAB through the MATLAB Engine API. MATLAB is temporarily changed into the case directory before `msparc(case_name)` is called. This makes relative `PSEUDO_POT` paths resolve from the case folder.

After a run the agent reports recognized files such as `.out`, `.static`, `.geopt`, `.cellopt`, `.aimd`, and `.restart`.

### 4. Output values are read locally

Common requests such as these do not require Gemini:

```text
What is the total energy of Fe2_case?
Show the Fermi level of Fe2_case.
What is the maximum force?
Show the pressure.
How many SCF iterations were used?
```

The response includes the source output file and line.

### 5. Faster hybrid architecture

Common actions use zero Gemini calls:

- list cases
- validate a case
- run an existing case
- list output files
- read common output quantities
- look up common Manual.tex parameter syntax/defaults
- standard mesh-convergence workflow when the base case and values are explicit

Gemini is used mainly for genuinely natural-language/ambiguous planning and for converting a new-case description into structured JSON. High-level local macro tools replace long chains of model calls.

### 6. Cleaner UI

Execution-step panels have been removed. The user sees the request, final result, and plots.

---

## Installation

### Python

Use a Python version supported by your MATLAB release. For MATLAB R2026a, Python 3.13 works with the R2026a engine package.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### MATLAB Engine

The engine package must match the installed MATLAB release. For MATLAB R2026a (26.1), install the matching 26.1 engine rather than the latest 26.2 package, for example:

```powershell
python -m pip install matlabengine==26.1.12
```

Verify:

```powershell
python -c "import matlab.engine; e=matlab.engine.start_matlab(); print(e.version()); e.quit()"
```

If you use another MATLAB release, install the engine version documented for that release.

### Gemini API key

PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

Or enter the key in the app sidebar.

---

## Folder configuration

Keep these separate:

```text
C:\...\msparc_agent_gemini\      <- this application
C:\...\MSPARC_Calculations\     <- writable simulation workspace
C:\...\M-SPARC-master\          <- official extracted M-SPARC repository
```

The M-SPARC path must contain:

```text
src/
doc/
tests/
psps/
```

---

## Start

```powershell
python -m streamlit run app.py
```

Use `python -m streamlit` so Streamlit uses the same virtual environment that contains the MATLAB Engine.

---

## Good tests

### Local manual knowledge

```text
What is the syntax and default value for mesh spacing?
```

### Existing case

```text
Validate Fe2_spin_gamma_ortho_vdWDF1.
```

```text
Run Fe2_spin_gamma_ortho_vdWDF1.
```

```text
What is the total energy of Fe2_spin_gamma_ortho_vdWDF1?
```

### Convergence study

```text
Using Fe2_spin_gamma_ortho_vdWDF1, perform a mesh convergence study for mesh spacings 0.20, 0.25, 0.30, 0.35 and 0.40. Run the cases, extract total energy and plot it against mesh spacing.
```

If you want a formal convergence decision, supply a tolerance:

```text
Use an energy tolerance of 1e-4 Ha.
```

### New case creation

Give enough scientific information, for example lattice/cell, grid resolution, species, coordinates and whether coordinates are Cartesian or fractional. The agent will ask for anything essential that is missing.

The structured generator supports a semantic specification like:

```json
{
  "inpt": {
    "LATVEC": [[1,0,0],[0,1,0],[0,0,1]],
    "LATVEC_SCALE": [2.84,2.84,2.84],
    "MESH_SPACING": 0.20,
    "BC": ["P","P","P"],
    "EXCHANGE_CORRELATION": "GGA_PBE"
  },
  "species": [
    {
      "element": "Fe",
      "coordinate_mode": "fractional",
      "coordinates": [[0,0,0],[0.5,0.5,0.5]],
      "pseudopotential": "auto"
    }
  ]
}
```

The user never needs to type JSON; this is the internal representation produced by the LLM.
