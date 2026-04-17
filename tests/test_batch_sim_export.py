import os
import re
import sys
import types
from pathlib import Path

import numpy as np


def _ensure_local_src_on_syspath():
    """Allow importing `pymars` from this repo without editable install."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


def _install_fake_runtime_modules(monkeypatch):
    """Install lightweight fake modules needed by pymars.__init__.main()."""

    # ---- fake jax ----
    class _FakeConfig:
        def update(self, *args, **kwargs):
            return None

    jax_mod = types.ModuleType("jax")
    jax_mod.config = _FakeConfig()
    jax_mod.devices = lambda: [object()]
    jax_mod.jit = lambda fn: fn

    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    # Allow `import jax.numpy as jnp` to behave like numpy in tests.
    monkeypatch.setitem(sys.modules, "jax.numpy", np)

    # ---- fake fennol module tree used by imports ----
    fennol_mod = types.ModuleType("fennol")
    fennol_utils_mod = types.ModuleType("fennol.utils")
    fennol_utils_mod.__path__ = []

    fennol_input_parser_mod = types.ModuleType("fennol.utils.input_parser")
    fennol_input_parser_mod.convert_dict_units = lambda params, _units: params

    fennol_io_mod = types.ModuleType("fennol.utils.io")
    fennol_io_mod.human_time_duration = lambda t: f"{float(t):.2f}s"
    fennol_io_mod.write_xyz_frame = lambda *args, **kwargs: None
    fennol_io_mod.last_xyz_frame = lambda *_args, **_kwargs: (
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=np.float32),
        "",
    )

    fennol_pt_mod = types.ModuleType("fennol.utils.periodic_table")
    fennol_pt_mod.PERIODIC_TABLE = ["", "H", "He", "Li", "Be", "B", "C", "N", "O"]
    fennol_pt_mod.PERIODIC_TABLE_REV_IDX = {"H": 1, "C": 6, "O": 8}
    masses = np.zeros(128, dtype=np.float32)
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
    monkeypatch.setitem(sys.modules, "fennol.utils", fennol_utils_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils.input_parser", fennol_input_parser_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils.io", fennol_io_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils.periodic_table", fennol_pt_mod)
    monkeypatch.setitem(sys.modules, "fennol.utils.atomic_units", fennol_au_mod)

    # ---- fake ase.data ----
    ase_mod = types.ModuleType("ase")
    ase_data_mod = types.ModuleType("ase.data")
    ase_data_mod.chemical_symbols = ["X"] * 128
    ase_data_mod.chemical_symbols[1] = "H"
    ase_data_mod.chemical_symbols[6] = "C"
    ase_data_mod.chemical_symbols[8] = "O"

    monkeypatch.setitem(sys.modules, "ase", ase_mod)
    monkeypatch.setitem(sys.modules, "ase.data", ase_data_mod)


def test_batch_sim_export_moves_outputs_and_rewrites_seed(tmp_path, monkeypatch):
    """Regression test for SIMXXXXX export: folders, moved files, and seed rewriting."""

    _ensure_local_src_on_syspath()
    _install_fake_runtime_modules(monkeypatch)

    # Import pymars after fake runtime modules are installed.
    import pymars
    import pymars.md as md

    batch_size = 2

    def _fake_initialize_collision_simulation(_simulation_parameters, verbose=True):
        n_atoms = 2
        coordinates = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        velocities = np.zeros_like(coordinates)
        accelerations = np.zeros_like(coordinates)
        species = np.array([1, 1], dtype=np.int32)
        masses = np.array([1.0, 1.0], dtype=np.float32)

        def integrate(initial_coordinates, initial_velocities, initial_accelerations, step=0, energy_output_file=None, energy_steps=100):
            # Create energy files at requested paths so export logic has artifacts to move.
            if energy_output_file is not None and step % energy_steps == 0:
                for b in range(batch_size):
                    with open(energy_output_file[b], "a", encoding="utf-8") as f:
                        f.write(f"step={step} traj={b}\n")

            energy_data = {
                "total_energies": np.array([1.0, 2.0], dtype=float),
                "potential_energies": np.array([0.3, 0.4], dtype=float),
                "kinetic_energies": np.array([0.7, 1.6], dtype=float),
                "temperatures": np.array([300.0, 310.0], dtype=float),
                "variances": None,
            }
            energies = np.array([1.0, 2.0], dtype=np.float32)
            return (
                initial_coordinates,
                initial_velocities,
                initial_accelerations,
                energies,
                energy_data,
                None,
            )

        return {
            "integrate": integrate,
            "coordinates": coordinates,
            "velocities": velocities,
            "accelerations": accelerations,
            "species": species,
            "masses": masses,
            "batch_size": batch_size,
            "dt": 0.001,  # ps (1 fs)
            "initial_energies": np.array([1.0, 2.0], dtype=np.float32),
        }

    monkeypatch.setattr(md, "initialize_collision_simulation", _fake_initialize_collision_simulation)

    # Create minimal input geometry and config.
    geometry_file = tmp_path / "geom.xyz"
    geometry_file.write_text(
        "2\ncomment\nH 0.0 0.0 0.0\nH 0.7 0.0 0.0\n",
        encoding="utf-8",
    )

    input_yaml = tmp_path / "input.yaml"
    input_yaml.write_text(
        "\n".join(
            [
                "general_parameters:",
                "  batch_size: 2",
                "  seed: 11,22",
                "  save_steps: 1",
                "  save_energy: 1",
                "  save_summary: 1",
                "input_parameters:",
                f"  initial_geometry: '{geometry_file}'",
                "  random_rotation: false",
                "dynamic_parameters:",
                "  dt_dyn: 1.0",
                "  step_dyn: 1",
                "output_details:",
                "  trajectory_file: none",
                "  energies_file: energies.out",
                "  summary_file: summary.out",
                "calculation_parameters:",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pymars", str(input_yaml)])

    # Run entrypoint.
    pymars.main()

    # Check SIM folders created.
    sim0 = tmp_path / "SIM00000"
    sim1 = tmp_path / "SIM00001"
    assert sim0.is_dir()
    assert sim1.is_dir()

    # Check per-trajectory outputs were moved and renamed to base names inside SIM folders.
    assert (sim0 / "energies.out").is_file()
    assert (sim1 / "energies.out").is_file()
    assert (sim0 / "summary.out").is_file()
    assert (sim1 / "summary.out").is_file()

    # Original per-trajectory temporary names should no longer exist at root after move.
    assert not (tmp_path / "energies_0.out").exists()
    assert not (tmp_path / "energies_1.out").exists()
    assert not (tmp_path / "summary_0.out").exists()
    assert not (tmp_path / "summary_1.out").exists()

    # Copied input YAML in each SIM folder should carry trajectory-specific seed.
    sim0_yaml = (sim0 / "input.yaml").read_text(encoding="utf-8")
    sim1_yaml = (sim1 / "input.yaml").read_text(encoding="utf-8")

    seed0 = re.search(r"^\s*seed\s*:\s*(\d+)\s*$", sim0_yaml, flags=re.MULTILINE)
    seed1 = re.search(r"^\s*seed\s*:\s*(\d+)\s*$", sim1_yaml, flags=re.MULTILINE)
    assert seed0 is not None and seed0.group(1) == "11"
    assert seed1 is not None and seed1.group(1) == "22"
