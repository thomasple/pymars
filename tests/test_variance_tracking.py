"""
Tests for track_variance functionality.

These tests mock model._energy_and_forces to return a known etot_ensemble_var value
and verify that:
  1. The Var(eV^2) column appears in the energies output file when track_variance=True.
  2. The written value matches the model-provided etot_ensemble_var.
  3. 'None' is written when the model does not return etot_ensemble_var.
  4. track_variance is disabled (with a warning) when batch_size > 1.
"""
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sim_params(track_variance: bool, batch_size: int, tmpdir: str, xyz_path: str) -> dict:
    return {
        "input_parameters": {
            "initial_geometry": str(xyz_path),
            "total_charge": 0,
            "random_rotation": False,
        },
        "general_parameters": {
            "temperature": 0.0,
            "batch_size": batch_size,
            "save_steps": 1,
            "save_energy": 1,
        },
        "calculation_parameters": {
            "model": str(Path(__file__).parent / "mock_model.fnx"),
            "device": "cpu",
        },
        "dynamic_parameters": {
            "dt_dyn": 1.0,
        },
        "output_details": {
            "trajectory_file": os.path.join(tmpdir, "traj.xyz"),
            "energies_file": os.path.join(tmpdir, "energies.out"),
            "track_variance": track_variance,
        },
        "thermostat_parameters": {
            "NVE_thermostat": True,
        },
    }


def _make_mock_model(etot_ensemble_var=None, n_atoms=21, batch_size=1):
    """Return a mock FENNIX model that returns a fixed aux dict."""
    model = MagicMock()
    model.energy_unit = "eV"

    # Preprocess and preproc_state
    model.preproc_state = MagicMock()
    model.preproc_state.copy.return_value = model.preproc_state

    def fake_preprocess(**kwargs):
        conformation = dict(kwargs)
        return conformation
    model.preprocess.side_effect = fake_preprocess

    # _energy_and_forces returns (energies, forces, aux)
    energies = np.zeros(batch_size, dtype=np.float32)
    forces = np.zeros((batch_size, n_atoms, 3), dtype=np.float32)

    if etot_ensemble_var is not None:
        aux = {"etot_ensemble_var": np.full(batch_size, etot_ensemble_var, dtype=np.float32)}
    else:
        aux = {}

    model._energy_and_forces.return_value = (energies, forces, aux)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVarianceTracking:
    """Tests for etot_ensemble_var extraction and output."""

    def test_variance_written_to_file_when_model_provides_it(self, tmp_path, test_data_dir):
        """When model returns etot_ensemble_var, the Var(eV^2) column is written correctly."""
        xyz = test_data_dir / "aspirin.xyz"
        if not xyz.exists():
            pytest.skip("aspirin.xyz not found in test data dir")

        sim_params = _build_sim_params(
            track_variance=True, batch_size=1,
            tmpdir=str(tmp_path), xyz_path=xyz
        )

        expected_var = 0.12345

        from pymars.md import initialize_collision_simulation

        mock_model = _make_mock_model(etot_ensemble_var=expected_var, n_atoms=21)

        with patch("fennol.FENNIX") as MockFENNIX:
            MockFENNIX.load.return_value = mock_model
            system = initialize_collision_simulation(sim_params, verbose=False)

        integrate = system["integrate"]
        coords = system["coordinates"]
        vels = system["velocities"]
        accs = system["accelerations"]
        energy_file = sim_params["output_details"]["energies_file"]

        # Run one step
        coords, vels, accs, energies, energy_data = integrate(
            coords, vels, accs,
            step=0,
            energy_output_file=[energy_file],
            energy_steps=1,
        )

        # Check energies file was created
        assert os.path.exists(energy_file), "energies.out not created"

        with open(energy_file) as f:
            lines = f.readlines()

        # Header should contain Var(eV^2)
        header = lines[0]
        assert "Var(eV^2)" in header, f"Var(eV^2) not in header: {header!r}"

        # Data line should have numeric variance
        data_line = lines[1].strip()
        fields = data_line.split()
        var_field = fields[-1]
        assert var_field != "None", "Expected a numeric variance, got None"
        assert abs(float(var_field) - expected_var) < 1e-4, (
            f"Variance mismatch: expected {expected_var}, got {var_field}"
        )

        # energy_data should expose variances
        assert energy_data is not None
        assert energy_data.get("variances") is not None
        assert abs(float(energy_data["variances"][0]) - expected_var) < 1e-4

    def test_variance_none_when_model_does_not_provide_it(self, tmp_path, test_data_dir):
        """When model does not return etot_ensemble_var, 'None' is written in Var column."""
        xyz = test_data_dir / "aspirin.xyz"
        if not xyz.exists():
            pytest.skip("aspirin.xyz not found in test data dir")

        sim_params = _build_sim_params(
            track_variance=True, batch_size=1,
            tmpdir=str(tmp_path), xyz_path=xyz
        )

        from pymars.md import initialize_collision_simulation
        mock_model = _make_mock_model(etot_ensemble_var=None, n_atoms=21)

        with patch("pymars.md.FENNIX") as MockFENNIX:
            MockFENNIX.load.return_value = mock_model
            system = initialize_collision_simulation(sim_params, verbose=False)

        integrate = system["integrate"]
        coords = system["coordinates"]
        vels = system["velocities"]
        accs = system["accelerations"]
        energy_file = sim_params["output_details"]["energies_file"]

        coords, vels, accs, energies, energy_data = integrate(
            coords, vels, accs,
            step=0,
            energy_output_file=[energy_file],
            energy_steps=1,
        )

        with open(energy_file) as f:
            lines = f.readlines()

        assert "Var(eV^2)" in lines[0], "Var(eV^2) column missing from header"
        data_line = lines[1].strip()
        fields = data_line.split()
        assert fields[-1] == "None", f"Expected 'None' in variance column, got {fields[-1]!r}"

    def test_variance_disabled_for_batch_gt1(self, tmp_path, test_data_dir, capsys):
        """When batch_size > 1 and track_variance=True, warning is printed and column not written."""
        xyz = test_data_dir / "aspirin.xyz"
        if not xyz.exists():
            pytest.skip("aspirin.xyz not found in test data dir")

        batch_size = 2
        sim_params = _build_sim_params(
            track_variance=True, batch_size=batch_size,
            tmpdir=str(tmp_path), xyz_path=xyz
        )

        from pymars.md import initialize_collision_simulation
        mock_model = _make_mock_model(etot_ensemble_var=0.5, n_atoms=21, batch_size=batch_size)

        with patch("pymars.md.FENNIX") as MockFENNIX:
            MockFENNIX.load.return_value = mock_model
            system = initialize_collision_simulation(sim_params, verbose=True)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out and "track_variance" in captured.out, (
            "Expected a track_variance warning for batch_size > 1"
        )

        # energies file should NOT have a Var column since track_variance was disabled
        integrate = system["integrate"]
        coords = system["coordinates"]
        vels = system["velocities"]
        accs = system["accelerations"]

        energy_files = [
            str(tmp_path / f"energies_{b}.out") for b in range(batch_size)
        ]

        coords, vels, accs, energies, energy_data = integrate(
            coords, vels, accs,
            step=0,
            energy_output_file=energy_files,
            energy_steps=1,
        )

        with open(energy_files[0]) as f:
            header = f.readline()

        assert "Var(eV^2)" not in header, (
            "Var(eV^2) column should not appear when track_variance was disabled for batch>1"
        )
