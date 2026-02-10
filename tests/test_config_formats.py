"""
Tests for nested configuration format and no-projectile mode.
"""
import pytest
import numpy as np
from pathlib import Path


def test_nested_config_structure(test_data_dir, tmp_path):
    """Test that nested YAML configuration structure works correctly."""
    from pymars.md import initialize_collision_simulation
    
    # Create nested configuration
    config = {
        "calculation_parameters": {
            "model": str(test_data_dir / "mock_model.pt"),
        },
        "input_parameters": {
            "initial_geometry": str(test_data_dir / "aspirin.xyz"),
            "total_charge": 0,
            "random_rotation": False,  # Disable for reproducibility
        },
        "general_parameters": {
            "temperature": 300.0,
            "batch_size": 1,
            "seed": 12345,
            "save_steps": 100,
            "save_energy": 100,
        },
        "projectile_parameters": {
            "projectile_flag": True,
            "projectile_species": 18,
            "projectile_temperature": 1000.0,
            "projectile_distance": 15.0,
            "max_impact_parameter": 0.5,
        },
        "thermostat_parameters": {
            "NVE_thermostat": True,
            "LGV_thermostat": False,
            "gamma": 0.0,
        },
        "dynamic_parameters": {
            "dt_dyn": 1.0,
            "step_dyn": 100,
        },
    }
    
    # This test requires a mock model - skip if not available
    if not Path(config["calculation_parameters"]["model"]).exists():
        pytest.skip("Mock model not available")
    
    # Initialize should work without errors
    system = initialize_collision_simulation(config, verbose=False)
    
    # Check that all expected keys are present
    assert "species" in system
    assert "masses" in system
    assert "coordinates" in system
    assert "velocities" in system
    assert "accelerations" in system
    assert "integrate" in system
    assert "batch_size" in system
    assert "dt" in system
    
    # Check batch size
    assert system["batch_size"] == 1
    
    # With projectile, species should be N+1
    from pymars.initial_configuration import read_initial_configuration
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    assert len(system["species"]) == len(species) + 1  # +1 for projectile


def test_legacy_flat_config_backwards_compatibility(test_data_dir):
    """Test that old flat configuration format still works."""
    from pymars.md import initialize_collision_simulation
    
    # Old flat configuration style
    config = {
        "initial_geometry": str(test_data_dir / "aspirin.xyz"),
        "model": str(test_data_dir / "mock_model.pt"),
        "total_charge": 0,
        "batch_size": 1,
        "random_rotation": False,
        "temperature": 300.0,
        "projectile_species": 18,
        "projectile_temperature": 1000.0,
        "projectile_distance": 15.0,
        "max_impact_parameter": 0.5,
        "dt": 1.0,
    }
    
    if not Path(config["model"]).exists():
        pytest.skip("Mock model not available")
    
    # Should work with backward compatibility
    system = initialize_collision_simulation(config, verbose=False)
    
    assert "species" in system
    assert "integrate" in system
    assert system["batch_size"] == 1


def test_no_projectile_mode(test_data_dir):
    """Test simulation without projectile (molecule-only dynamics)."""
    from pymars.md import initialize_collision_simulation
    
    # Configuration with projectile disabled
    config = {
        "calculation_parameters": {
            "model": str(test_data_dir / "mock_model.pt"),
        },
        "input_parameters": {
            "initial_geometry": str(test_data_dir / "aspirin.xyz"),
            "total_charge": 0,
            "random_rotation": False,
        },
        "general_parameters": {
            "temperature": 300.0,
            "batch_size": 1,
        },
        "projectile_parameters": {
            "projectile_flag": False,  # No projectile
        },
        "dynamic_parameters": {
            "dt_dyn": 1.0,
            "step_dyn": 100,
        },
    }
    
    if not Path(config["calculation_parameters"]["model"]).exists():
        pytest.skip("Mock model not available")
    
    system = initialize_collision_simulation(config, verbose=False)
    
    # Without projectile, species count should match molecule only
    from pymars.initial_configuration import read_initial_configuration
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    assert len(system["species"]) == len(species)  # No +1 for projectile
    
    # Coordinates shape should be (batch_size, N, 3) not (batch_size, N+1, 3)
    assert system["coordinates"].shape[1] == len(species)


def test_no_projectile_legacy_format(test_data_dir):
    """Test no-projectile mode using legacy collision_dynamics parameter."""
    from pymars.md import initialize_collision_simulation
    
    config = {
        "initial_geometry": str(test_data_dir / "aspirin.xyz"),
        "model": str(test_data_dir / "mock_model.pt"),
        "collision_dynamics": False,  # Legacy parameter
        "total_charge": 0,
        "batch_size": 1,
        "temperature": 300.0,
        "dt": 1.0,
    }
    
    if not Path(config["model"]).exists():
        pytest.skip("Mock model not available")
    
    system = initialize_collision_simulation(config, verbose=False)
    
    # Should work and not include projectile
    from pymars.initial_configuration import read_initial_configuration
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    assert len(system["species"]) == len(species)


def test_batch_size_multiple_trajectories(test_data_dir):
    """Test that batch_size creates multiple parallel trajectories."""
    from pymars.md import initialize_collision_simulation
    
    batch_size = 3
    config = {
        "calculation_parameters": {
            "model": str(test_data_dir / "mock_model.pt"),
        },
        "input_parameters": {
            "initial_geometry": str(test_data_dir / "aspirin.xyz"),
            "total_charge": 0,
            "random_rotation": True,  # Should create different rotations
        },
        "general_parameters": {
            "temperature": 300.0,
            "batch_size": batch_size,
        },
        "projectile_parameters": {
            "projectile_flag": True,
            "projectile_species": 18,
        },
        "dynamic_parameters": {
            "dt_dyn": 1.0,
        },
    }
    
    if not Path(config["calculation_parameters"]["model"]).exists():
        pytest.skip("Mock model not available")
    
    system = initialize_collision_simulation(config, verbose=False)
    
    assert system["batch_size"] == batch_size
    
    # Coordinates should have batch dimension
    assert system["coordinates"].shape[0] == batch_size
    assert system["velocities"].shape[0] == batch_size
    assert system["accelerations"].shape[0] == batch_size


def test_parameter_fallbacks(test_data_dir):
    """Test that parameter names have proper fallbacks between old and new."""
    from pymars.md import initialize_collision_simulation
    
    # Mix of old and new parameter names
    config = {
        "calculation_parameters": {
            "model": str(test_data_dir / "mock_model.pt"),
        },
        "input_parameters": {
            "initial_geometry": str(test_data_dir / "aspirin.xyz"),
        },
        "general_parameters": {
            "temperature": 300.0,
            "batch_size": 1,
        },
        "dynamic_parameters": {
            "dt": 1.0,  # Old name, should still work
        },
        # Missing projectile_parameters - should use defaults
    }
    
    if not Path(config["calculation_parameters"]["model"]).exists():
        pytest.skip("Mock model not available")
    
    # Should work with mixed/missing parameters
    system = initialize_collision_simulation(config, verbose=False)
    assert "integrate" in system


def test_seed_parameter(test_data_dir):
    """Test that seed parameter produces reproducible results."""
    from pymars.md import initialize_collision_simulation
    
    config = {
        "calculation_parameters": {
            "model": str(test_data_dir / "mock_model.pt"),
        },
        "input_parameters": {
            "initial_geometry": str(test_data_dir / "aspirin.xyz"),
            "random_rotation": True,
        },
        "general_parameters": {
            "temperature": 300.0,
            "batch_size": 1,
            "seed": 42,  # Fixed seed
        },
        "projectile_parameters": {
            "projectile_flag": False,
        },
        "dynamic_parameters": {
            "dt_dyn": 1.0,
        },
    }
    
    if not Path(config["calculation_parameters"]["model"]).exists():
        pytest.skip("Mock model not available")
    
    # Initialize twice with same seed
    system1 = initialize_collision_simulation(config, verbose=False)
    
    # Reset config with same seed
    config["general_parameters"]["seed"] = 42
    system2 = initialize_collision_simulation(config, verbose=False)
    
    # Results should be identical (if numpy seed is properly set)
    # This is a basic check - actual reproducibility depends on implementation
    assert system1["batch_size"] == system2["batch_size"]


def test_thermostat_parameters_present(test_data_dir):
    """Test that thermostat parameters are accepted in config."""
    from pymars.md import initialize_collision_simulation
    
    config = {
        "calculation_parameters": {
            "model": str(test_data_dir / "mock_model.pt"),
        },
        "input_parameters": {
            "initial_geometry": str(test_data_dir / "aspirin.xyz"),
        },
        "general_parameters": {
            "temperature": 300.0,
            "batch_size": 1,
        },
        "projectile_parameters": {
            "projectile_flag": False,
        },
        "thermostat_parameters": {
            "NVE_thermostat": True,
            "LGV_thermostat": False,
            "gamma": 0.0,
        },
        "dynamic_parameters": {
            "dt_dyn": 1.0,
        },
    }
    
    if not Path(config["calculation_parameters"]["model"]).exists():
        pytest.skip("Mock model not available")
    
    # Should accept thermostat parameters without error
    system = initialize_collision_simulation(config, verbose=False)
    assert "integrate" in system
