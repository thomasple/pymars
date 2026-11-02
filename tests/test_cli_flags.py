import os
import subprocess
import sys
import types
from pathlib import Path

import pytest


def _ensure_local_src_on_syspath():
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


def _install_fake_runtime_modules(monkeypatch):
    """Install minimal fake modules needed by pymars and phoboos imports."""

    class _FakeConfig:
        def update(self, *args, **kwargs):
            return None

    jax_mod = types.ModuleType("jax")
    jax_mod.__version__ = "0.4.0"
    jax_mod.config = _FakeConfig()
    jax_mod.devices = lambda: [object()]
    jax_mod.jit = lambda fn: fn
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", __import__("numpy"))

    fennol_mod = types.ModuleType("fennol")
    fennol_mod.__path__ = []
    fennol_utils_mod = types.ModuleType("fennol.utils")
    fennol_utils_mod.__path__ = []
    fennol_models_mod = types.ModuleType("fennol.models")
    fennol_models_mod.FENNIX = object
    fennol_ase_mod = types.ModuleType("fennol.ase")
    fennol_ase_mod.FENNIXCalculator = object

    fennol_input_parser_mod = types.ModuleType("fennol.utils.input_parser")
    fennol_input_parser_mod.convert_dict_units = lambda params, _units: params

    fennol_io_mod = types.ModuleType("fennol.utils.io")
    fennol_io_mod.human_time_duration = lambda t: f"{float(t):.2f}s"
    fennol_io_mod.write_xyz_frame = lambda *args, **kwargs: None
    fennol_io_mod.last_xyz_frame = lambda *_args, **_kwargs: (
        ["H", "H"],
        __import__("numpy").array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=__import__("numpy").float32),
        "",
    )

    fennol_pt_mod = types.ModuleType("fennol.utils.periodic_table")
    fennol_pt_mod.PERIODIC_TABLE = ["", "H", "He", "Li", "Be", "B", "C", "N", "O"]
    fennol_pt_mod.PERIODIC_TABLE_REV_IDX = {"H": 1, "C": 6, "O": 8}
    masses = __import__("numpy").zeros(128, dtype=__import__("numpy").float32)
    masses[1] = 1.0
    masses[6] = 12.0
    masses[8] = 16.0
    fennol_pt_mod.ATOMIC_MASSES = masses

    fennol_au_mod = types.ModuleType("fennol.utils.atomic_units")

    class _FakeUnitSystem:
        def __init__(self, *args, **kwargs):
            self.NS = 1.0
            self.PS = 1.0
            self.K_B = 1.0
            self.DA = 1.0

        def get_multiplier(self, _unit):
            return 1.0

    fennol_au_mod.UnitSystem = _FakeUnitSystem

    monkeypatch.setitem(sys.modules, "fennol", fennol_mod)
    monkeypatch.setitem(sys.modules, "fennol.models", fennol_models_mod)
    monkeypatch.setitem(sys.modules, "fennol.ase", fennol_ase_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils", fennol_utils_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils.input_parser", fennol_input_parser_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils.io", fennol_io_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils.periodic_table", fennol_pt_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils.atomic_units", fennol_au_mod)

    ase_mod = types.ModuleType("ase")
    ase_data_mod = types.ModuleType("ase.data")
    ase_data_mod.chemical_symbols = ["X"] * 128
    ase_data_mod.chemical_symbols[1] = "H"
    ase_data_mod.chemical_symbols[6] = "C"
    ase_data_mod.chemical_symbols[8] = "O"
    monkeypatch.setitem(sys.modules, "ase", ase_mod)
    monkeypatch.setitem(sys.modules, "ase.data", ase_data_mod)


def test_pymars_version_flag_prints_version(monkeypatch, capsys):
    _ensure_local_src_on_syspath()
    _install_fake_runtime_modules(monkeypatch)

    import pymars

    monkeypatch.setattr(sys, "argv", ["pymars", "--version"])
    with pytest.raises(SystemExit) as exc:
        pymars.main()

    assert exc.value.code == 0
    captured = capsys.readouterr().out
    assert "pymars version" in captured


def test_pymars_test_flag_runs_pytest(monkeypatch):
    _ensure_local_src_on_syspath()
    _install_fake_runtime_modules(monkeypatch)

    import pymars

    calls = {}

    class DummyResult:
        returncode = 7

    def fake_run(cmd, cwd, check=False):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["check"] = check
        return DummyResult()

    monkeypatch.setattr(sys, "argv", ["pymars", "--test"])
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        pymars.main()

    assert exc.value.code == 7
    assert calls["cmd"] == [sys.executable, "-m", "pytest", "-q"]
    assert calls["check"] is False


def test_phoboos_spt_command_dispatches(monkeypatch, tmp_path):
    _ensure_local_src_on_syspath()
    _install_fake_runtime_modules(monkeypatch)

    import pymars.phoboos as phoboos

    input_yaml = tmp_path / "input.yaml"
    geom = tmp_path / "geom.xyz"
    geom.write_text("2\ncomment\nH 0 0 0\nH 0 0 1\n", encoding="utf-8")
    input_yaml.write_text(
        """
input_parameters:
  initial_geometry: geom.xyz
  total_charge: 0
calculation_parameters:
  model: model.fnx
output_parameters:
  verbose: false
""".strip(),
        encoding="utf-8",
    )

    calls = {}

    def fake_run_spt(xyz, model_file, total_charge, verbose):
        calls["spt"] = (xyz, model_file, total_charge, verbose)

    monkeypatch.setattr(phoboos, "__file__", str(tmp_path / "__init__.py"), raising=False)
    monkeypatch.setattr(sys, "argv", ["phoboos", "spt", str(input_yaml)])
    monkeypatch.setattr("pymars.phoboos.singlepoint.run_spt", fake_run_spt)

    with pytest.raises(SystemExit) as exc:
        phoboos.main()

    assert exc.value.code == 0
    assert calls["spt"][0] == str(geom)
    assert calls["spt"][2] == 0


def test_phoboos_opt_command_dispatches(monkeypatch, tmp_path):
    _ensure_local_src_on_syspath()
    _install_fake_runtime_modules(monkeypatch)

    import pymars.phoboos as phoboos

    input_yaml = tmp_path / "input.yaml"
    geom = tmp_path / "geom.xyz"
    geom.write_text("2\ncomment\nH 0 0 0\nH 0 0 1\n", encoding="utf-8")
    input_yaml.write_text(
        """
input_parameters:
  initial_geometry: geom.xyz
  total_charge: 0
calculation_parameters:
  model: model.fnx
optimization_parameters:
  tolerance: 1e-2
  dx_max: 0.2
  dt_dyn: 2.0
  max_steps: 10
  save_steps: -1
output_parameters:
  final_singlepoint: false
  frequency_calculation: false
""".strip(),
        encoding="utf-8",
    )

    called = {}

    def fake_run_opt(**kwargs):
        called["kwargs"] = kwargs
        return ("opt.xyz", 1.0, None, None)

    monkeypatch.setattr(sys, "argv", ["phoboos", "opt", str(input_yaml)])
    monkeypatch.setattr("pymars.phoboos.optiminimize.run_opt", fake_run_opt)

    with pytest.raises(SystemExit) as exc:
        phoboos.main()

    assert exc.value.code == 0
    assert called["kwargs"]["xyz_file"] == str(geom)


def test_phoboos_unknown_command_exits_1(monkeypatch, tmp_path):
    _ensure_local_src_on_syspath()
    _install_fake_runtime_modules(monkeypatch)

    import pymars.phoboos as phoboos

    input_yaml = tmp_path / "input.yaml"
    input_yaml.write_text(
        """
input_parameters:
  initial_geometry: geom.xyz
calculation_parameters:
  model: model.fnx
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["phoboos", "unknown", str(input_yaml)])

    with pytest.raises(SystemExit) as exc:
        phoboos.main()

    assert exc.value.code == 2
