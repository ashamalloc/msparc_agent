from pathlib import Path

from analysis import convergence_report, extract_numeric_matches
from workspace import Workspace


def test_inpt_read_and_update(tmp_path: Path):
    (tmp_path / "sample.inpt").write_text("BC: 2\nMESH_SPACING: 0.30\n", encoding="utf-8")
    (tmp_path / "sample.ion").write_text("ATOM_TYPE: Si\n", encoding="utf-8")
    workspace = Workspace(tmp_path)

    assert workspace.case_files("sample")["valid_pair"] is True
    parameters = workspace.read_inpt_parameters("sample")
    assert parameters[1]["parameter"] == "MESH_SPACING"

    result = workspace.set_inpt_parameter("sample", "MESH_SPACING", "0.25")
    assert result["previous"] == "MESH_SPACING: 0.30"
    assert "MESH_SPACING: 0.25" in (tmp_path / "sample.inpt").read_text(encoding="utf-8")


def test_extract_numeric_matches(tmp_path: Path):
    path = tmp_path / "sample.out"
    path.write_text("Total energy: -10.25 Ha\nTotal energy: -10.30 Ha\n", encoding="utf-8")
    matches = extract_numeric_matches(path, "Total energy")
    assert matches[-1]["numbers"] == [-10.30]


def test_convergence_report():
    report = convergence_report([0.4, 0.3, 0.2, 0.1], [-10.0, -10.1, -10.11, -10.111], 0.02)
    assert report["converged_from_x"] == 0.3


def test_discover_cases_and_refout(tmp_path: Path):
    (tmp_path / "good.inpt").write_text("BC: 2\n", encoding="utf-8")
    (tmp_path / "good.ion").write_text("ATOM_TYPE: Si\n", encoding="utf-8")
    (tmp_path / "good.refout").write_text("Total energy: -1.0\n", encoding="utf-8")
    (tmp_path / "incomplete.inpt").write_text("BC: 2\n", encoding="utf-8")

    workspace = Workspace(tmp_path)
    cases = workspace.discover_cases()

    assert len(cases) == 2
    good = next(case for case in cases if case["case_stem"] == "good")
    assert good["valid_pair"] is True
    assert "good.refout" in good["outputs"]

    incomplete = next(case for case in cases if case["case_stem"] == "incomplete")
    assert incomplete["valid_pair"] is False


def test_duplicate_case_repairs_relative_psp_path(tmp_path: Path):
    psps = tmp_path / "psps"
    psps.mkdir()
    (psps / "Fe.psp8").write_text("dummy", encoding="utf-8")

    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "Fe.inpt").write_text("MESH_SPACING: 0.30\n", encoding="utf-8")
    (cases / "Fe.ion").write_text("ATOM_TYPE: Fe\nPSEUDO_POT: ../psps/Fe.psp8\n", encoding="utf-8")

    workspace = Workspace(tmp_path)
    result = workspace.duplicate_case("cases/Fe", "cases/sweeps/Fe_mesh")

    copied_ion = (tmp_path / "cases" / "sweeps" / "Fe_mesh.ion").read_text(encoding="utf-8")
    assert "../../psps/Fe.psp8" in copied_ion.replace("\\", "/")
    assert result["pseudopotential_path_changes"]
    refs = workspace.pseudopotential_references("cases/sweeps/Fe_mesh")
    assert refs and refs[0]["exists"] is True


def test_reference_index_resolves_official_keyword(tmp_path: Path):
    from reference import MSPARCReference

    (tmp_path / "src").mkdir()
    (tmp_path / "doc").mkdir()
    test_dir = tmp_path / "tests" / "Example"
    test_dir.mkdir(parents=True)

    (tmp_path / "doc" / "manual.md").write_text(
        "MESH_SPACING: controls the real-space mesh spacing.\n",
        encoding="utf-8",
    )
    (test_dir / "sample.inpt").write_text("MESH_SPACING: 0.30\nBC: 2\n", encoding="utf-8")
    (test_dir / "sample.ion").write_text("ATOM_TYPE: Fe\n", encoding="utf-8")

    reference = MSPARCReference(tmp_path)
    resolved = reference.resolve_input_parameter("mesh spacing")

    assert reference.known_inpt_keyword("MESH_SPACING")
    assert resolved["candidates"]
    assert resolved["candidates"][0]["keyword"] == "MESH_SPACING"


def test_reference_search_matches_spaces_to_underscores(tmp_path: Path):
    from reference import MSPARCReference

    (tmp_path / "src").mkdir()
    (tmp_path / "doc").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "doc" / "manual.txt").write_text("MESH_SPACING: real space grid control\n", encoding="utf-8")

    reference = MSPARCReference(tmp_path)
    hits = reference.search("mesh spacing")
    assert hits
    assert "MESH_SPACING" in hits[0]["text"]


def test_reuse_existing_case_repairs_old_broken_psp_path(tmp_path: Path):
    psps = tmp_path / "psps"
    psps.mkdir()
    (psps / "Fe.psp8").write_text("dummy", encoding="utf-8")

    cases = tmp_path / "cases"
    sweep = cases / "sweeps"
    sweep.mkdir(parents=True)
    (cases / "Fe.inpt").write_text("MESH_SPACING: 0.30\n", encoding="utf-8")
    (cases / "Fe.ion").write_text("PSEUDO_POT: ../psps/Fe.psp8\n", encoding="utf-8")

    # Simulate a v1 copy that retained the old relative path and is therefore broken.
    (sweep / "Fe_mesh.inpt").write_text("MESH_SPACING: 0.20\n", encoding="utf-8")
    (sweep / "Fe_mesh.ion").write_text("PSEUDO_POT: ../psps/Fe.psp8\n", encoding="utf-8")

    workspace = Workspace(tmp_path)
    before = workspace.pseudopotential_references("cases/sweeps/Fe_mesh")
    assert before[0]["exists"] is False

    result = workspace.duplicate_case("cases/Fe", "cases/sweeps/Fe_mesh", reuse_existing=True)
    after = workspace.pseudopotential_references("cases/sweeps/Fe_mesh")

    assert result["reused_existing"] is True
    assert result["pseudopotential_path_changes"]
    assert after[0]["exists"] is True


def test_manual_parameter_help_extracts_type_example_and_description(tmp_path: Path):
    from reference import MSPARCReference

    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "doc").mkdir()
    manual = r'''\begin{frame}[allowframebreaks]{\texttt{MESH\_SPACING}} \label{MESH_SPACING}
\begin{block}{Type}
Double
\end{block}
\begin{block}{Unit}
Bohr
\end{block}
\begin{block}{Example}
\texttt{MESH\_SPACING}: 0.4
\end{block}
\begin{block}{Description}
Mesh spacing of the finite-difference grid.
\end{block}
\end{frame}
'''
    (tmp_path / "doc" / "Manual.tex").write_text(manual, encoding="utf-8")

    reference = MSPARCReference(tmp_path)
    help_info = reference.manual_parameter_help("MESH_SPACING")

    assert help_info["found"] is True
    assert help_info["type"] == "Double"
    assert help_info["unit"] == "Bohr"
    assert "MESH_SPACING" in help_info["example"]
    assert "finite-difference" in help_info["description"]


def _make_fake_msparc_repo(root: Path):
    (root / "src").mkdir(parents=True)
    (root / "doc").mkdir()
    (root / "psps").mkdir()
    test_dir = root / "tests" / "Fe" / "standard"
    test_dir.mkdir(parents=True)
    manual = r'''
\begin{frame}[allowframebreaks]{\texttt{LATVEC}} \label{LATVEC}
\begin{block}{Type}Double array\end{block}
\begin{block}{Default}None\end{block}
\begin{block}{Description}Lattice vectors.\end{block}
\end{frame}
\begin{frame}[allowframebreaks]{\texttt{LATVEC\_SCALE}} \label{LATVEC_SCALE}
\begin{block}{Type}Double array\end{block}
\begin{block}{Default}1 1 1\end{block}
\begin{block}{Description}Lattice vector scale.\end{block}
\end{frame}
\begin{frame}[allowframebreaks]{\texttt{MESH\_SPACING}} \label{MESH_SPACING}
\begin{block}{Type}Double\end{block}
\begin{block}{Default}None\end{block}
\begin{block}{Unit}Bohr\end{block}
\begin{block}{Example}\texttt{MESH\_SPACING}: 0.4\end{block}
\begin{block}{Description}Mesh spacing of the finite-difference grid.\end{block}
\begin{block}{Remark}MESH_SPACING, FD_GRID, ECUT cannot be specified simultaneously.\end{block}
\end{frame}
\begin{frame}[allowframebreaks]{\texttt{FD\_GRID}} \label{FD_GRID}
\begin{block}{Type}Integer array\end{block}
\begin{block}{Default}None\end{block}
\begin{block}{Description}Finite difference grid.\end{block}
\end{frame}
\begin{frame}[allowframebreaks]{\texttt{ECUT}} \label{ECUT}
\begin{block}{Type}Double\end{block}
\begin{block}{Default}None\end{block}
\begin{block}{Description}Energy cutoff.\end{block}
\end{frame}
\begin{frame}[allowframebreaks]{\texttt{BC}} \label{BC}
\begin{block}{Type}Character array\end{block}
\begin{block}{Default}P P P\end{block}
\begin{block}{Description}Boundary conditions.\end{block}
\end{frame}
\begin{frame}[allowframebreaks]{\texttt{EXCHANGE\_CORRELATION}} \label{EXCHANGE_CORRELATION}
\begin{block}{Type}Character\end{block}
\begin{block}{Default}GGA_PBE\end{block}
\begin{block}{Description}Exchange correlation functional.\end{block}
\end{frame}
'''
    (root / "doc" / "Manual.tex").write_text(manual, encoding="utf-8")
    (root / "psps" / "26_Fe_test_pbe_n_v1.0.psp8").write_text("dummy", encoding="utf-8")
    (test_dir / "Fe.inpt").write_text(
        "LATVEC:\n1 0 0\n0 1 0\n0 0 1\nLATVEC_SCALE: 2.8 2.8 2.8\nMESH_SPACING: 0.2\nBC: P P P\nEXCHANGE_CORRELATION: GGA_PBE\n",
        encoding="utf-8",
    )
    (test_dir / "Fe.ion").write_text(
        "ATOM_TYPE: Fe\nPSEUDO_POT: ../../../psps/26_Fe_test_pbe_n_v1.0.psp8\nN_TYPE_ATOM: 1\nCOORD_FRAC:\n0 0 0\n",
        encoding="utf-8",
    )


def test_case_builder_generates_valid_syntax_from_structured_spec(tmp_path: Path):
    import json
    from case_builder import MSPARCCaseBuilder
    from reference import MSPARCReference
    from workspace import Workspace

    repo = tmp_path / "repo"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _make_fake_msparc_repo(repo)
    reference = MSPARCReference(repo)
    builder = MSPARCCaseBuilder(Workspace(workspace_root), reference, repo)
    spec = {
        "inpt": {
            "LATVEC": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "LATVEC_SCALE": [2.8, 2.8, 2.8],
            "MESH_SPACING": 0.2,
            "BC": ["P", "P", "P"],
            "EXCHANGE_CORRELATION": "GGA_PBE",
        },
        "species": [{
            "element": "Fe",
            "coordinate_mode": "fractional",
            "coordinates": [[0, 0, 0], [0.5, 0.5, 0.5]],
            "pseudopotential": "auto",
        }],
    }
    result = builder.create_case("generated/Fe2", json.dumps(spec))
    assert result["success"] is True
    inpt = (workspace_root / "generated" / "Fe2.inpt").read_text(encoding="utf-8")
    ion = (workspace_root / "generated" / "Fe2.ion").read_text(encoding="utf-8")
    assert "LATVEC:\n1 0 0\n0 1 0\n0 0 1" in inpt
    assert "MESH_SPACING: 0.2" in inpt
    assert "ATOM_TYPE: Fe" in ion
    assert "N_TYPE_ATOM: 2" in ion
    assert "COORD_FRAC:" in ion
    assert "26_Fe_test_pbe_n_v1.0.psp8" in ion


def test_case_builder_rejects_conflicting_grid_controls(tmp_path: Path):
    import json
    from case_builder import MSPARCCaseBuilder
    from reference import MSPARCReference
    from workspace import Workspace

    repo = tmp_path / "repo"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _make_fake_msparc_repo(repo)
    builder = MSPARCCaseBuilder(Workspace(workspace_root), MSPARCReference(repo), repo)
    spec = {
        "inpt": {
            "LATVEC": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "MESH_SPACING": 0.2,
            "ECUT": 30,
        },
        "species": [{"element": "Fe", "coordinate_mode": "fractional", "coordinates": [[0, 0, 0]]}],
    }
    result = builder.create_case("bad", json.dumps(spec))
    assert result["success"] is False
    assert "exactly one" in " ".join(result["validation"]["errors"]).lower()


def test_output_reader_returns_final_total_energy(tmp_path: Path):
    from output_reader import MSPARCOutputReader
    from workspace import Workspace

    (tmp_path / "case.inpt").write_text("LATVEC:\n1 0 0\n0 1 0\n0 0 1\nMESH_SPACING: 0.2\n", encoding="utf-8")
    (tmp_path / "case.ion").write_text("ATOM_TYPE: Fe\n", encoding="utf-8")
    (tmp_path / "case.out").write_text(
        "Total free energy                  : -2.0000000000E+02 (Ha)\n"
        "Fermi level                        :  1.2500000000E+00 (Ha)\n",
        encoding="utf-8",
    )
    reader = MSPARCOutputReader(Workspace(tmp_path))
    result = reader.query("case", "total energy")
    assert result["success"] is True
    assert result["value"] == -200.0
    assert result["source_file"] == "case.out"


def test_remove_inpt_parameter(tmp_path: Path):
    workspace = Workspace(tmp_path)
    (tmp_path / "case.inpt").write_text("LATVEC:\n1 0 0\n0 1 0\n0 0 1\nFD_GRID: 20 20 20\nMESH_SPACING: 0.2\n", encoding="utf-8")
    result = workspace.remove_inpt_parameter("case", "FD_GRID")
    assert result["count"] == 1
    text = (tmp_path / "case.inpt").read_text(encoding="utf-8")
    assert "FD_GRID" not in text
    assert "MESH_SPACING" in text


def test_manual_reference_excludes_ion_only_from_inpt_keywords(tmp_path: Path):
    from reference import MSPARCReference

    (tmp_path / "src").mkdir()
    (tmp_path / "doc").mkdir()
    test_dir = tmp_path / "tests" / "Example"
    test_dir.mkdir(parents=True)
    (tmp_path / "doc" / "Manual.tex").write_text(
        r'''\begin{frame}{\texttt{MESH\_SPACING}} \label{MESH_SPACING}
\begin{block}{Type}Double\end{block}\end{frame}
\begin{frame}{\texttt{ATOM\_TYPE}} \label{ATOM_TYPE}
\begin{block}{Type}Character\end{block}\end{frame}''', encoding="utf-8"
    )
    (test_dir / "x.inpt").write_text("MESH_SPACING: 0.2\n", encoding="utf-8")
    (test_dir / "x.ion").write_text("ATOM_TYPE: Fe\n", encoding="utf-8")
    reference = MSPARCReference(tmp_path)
    assert reference.known_inpt_keyword("MESH_SPACING")
    assert not reference.known_inpt_keyword("ATOM_TYPE")
