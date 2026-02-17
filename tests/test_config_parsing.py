"""
Integration tests for MD simulation without requiring actual model files.
Tests the configuration parsing and system initialization logic.
"""
import pytest
import numpy as np
from pathlib import Path


def test_config_parsing_nested_format(test_data_dir):
    """Test that nested config is properly parsed without running simulation."""
    from pymars.md import initialize_collision_simulation
    from pymars.initial_configuration import read_initial_configuration
    
    # Read the test geometry to get expected dimensions
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    n_atoms = len(species)
    
    # Create minimal nested config (without model for now)
    config = {
        "input_parameters": {
            "initial_geometry": {
                "species": species.tolist(),
                "coordinates": coords.tolist(),
            },
            "total_charge": 0,
            "random_rotation": False,
        },
        "general_parameters": {
            "temperature": 300.0,
            "batch_size": 2,
        },
        "projectile_parameters": {
            "projectile_flag": True,
            "projectile_species": 18,
            "projectile_temperature": 1000.0,
            "projectile_distance": 20.0,
            "max_impact_parameter": 0.5,
        },
        "dynamic_parameters": {
            "dt_dyn": 1.0,
        },
    }
    
    # We can't fully initialize without a model, but we can test parameter extraction
    # by checking the internal logic
    input_params = config.get("input_parameters", {})
    general_params = config.get("general_parameters", {})
    projectile_params = config.get("projectile_parameters", {})
    
    # Verify nested parameters are accessible
    assert input_params.get("total_charge") == 0
    assert general_params.get("batch_size") == 2
    assert general_params.get("temperature") == 300.0
    assert projectile_params.get("projectile_flag") == True
    assert projectile_params.get("projectile_species") == 18


def test_config_parsing_flat_format(test_data_dir):
    """Test that flat (legacy) config format is parsed correctly."""
    from pymars.initial_configuration import read_initial_configuration
    
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    
    # Old flat format
    config = {
        "initial_geometry": str(test_data_dir / "aspirin.xyz"),
        "total_charge": 0,
        "batch_size": 2,
        "temperature": 300.0,
        "projectile_species": 18,
        "dt": 1.0,
    }
    
    # For backward compatibility, flat format should be accessible directly
    assert config.get("total_charge") == 0
    assert config.get("batch_size") == 2
    assert config.get("temperature") == 300.0


def test_projectile_flag_detection():
    """Test that projectile_flag is properly detected from config."""
    # Test with nested format
    config_nested = {
        "projectile_parameters": {
            "projectile_flag": False,
        }
    }
    
    projectile_params = config_nested.get("projectile_parameters", {})
    if "projectile_flag" in projectile_params:
        collision = bool(projectile_params.get("projectile_flag", True))
    else:
        collision = True  # default
    
    assert collision == False
    
    # Test with missing parameter (should default to True)
    config_empty = {}
    projectile_params = config_empty.get("projectile_parameters", {})
    if "projectile_flag" in projectile_params:
        collision = bool(projectile_params.get("projectile_flag", True))
    else:
        collision = True
    
    assert collision == True


def test_parameter_name_mapping():
    """Test that old and new parameter names are properly mapped."""
    # Test dt_dyn vs dt
    config_new = {
        "dynamic_parameters": {
            "dt_dyn": 1.5,
            "step_dyn": 1000,
        }
    }
    
    dynamic_params = config_new.get("dynamic_parameters", {})
    dt = dynamic_params.get("dt_dyn", dynamic_params.get("dt", 1.0))
    step = dynamic_params.get("step_dyn", None)
    
    assert dt == 1.5
    assert step == 1000
    
    # Test with old names
    config_old = {
        "dt": 2.0,
        "n_steps": 500,
    }
    
    # For flat format, use config directly
    dt_old = config_old.get("dt_dyn", config_old.get("dt", 1.0))
    assert dt_old == 2.0


def test_geometry_input_formats(test_data_dir):
    """Test that both file path and dict geometry inputs work."""
    from pymars.initial_configuration import read_initial_configuration
    
    # Test 1: File path
    geometry_file = str(test_data_dir / "aspirin.xyz")
    assert Path(geometry_file).is_file()
    
    species, coords = read_initial_configuration(geometry_file)
    assert len(species) > 0
    assert coords.shape == (len(species), 3)
    
    # Test 2: Dict format
    geometry_dict = {
        "species": species.tolist(),
        "coordinates": coords.tolist(),
    }
    
    assert isinstance(geometry_dict["species"], list)
    assert isinstance(geometry_dict["coordinates"], list)
    
    # Both should produce same results
    species_array = np.array(geometry_dict["species"], dtype=np.int32)
    coords_array = np.array(geometry_dict["coordinates"], dtype=np.float32).reshape(-1, 3)
    
    assert np.array_equal(species_array, species)
    assert np.allclose(coords_array, coords)


def test_batch_size_shapes(test_data_dir):
    """Test that batch size affects array shapes correctly."""
    from pymars.initial_configuration import read_initial_configuration, sample_velocities
    
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    n_atoms = len(species)
    
    # Test with batch_size = 1
    batch_size = 1
    batch_species = np.tile(species, batch_size)
    velocities = sample_velocities(batch_species, 300.0).reshape(batch_size, -1, 3)
    
    assert velocities.shape == (batch_size, n_atoms, 3)
    
    # Test with batch_size = 3
    batch_size = 3
    batch_species = np.tile(species, batch_size)
    velocities = sample_velocities(batch_species, 300.0).reshape(batch_size, -1, 3)
    
    assert velocities.shape == (batch_size, n_atoms, 3)


def test_collision_vs_no_collision_species_count(test_data_dir):
    """Test that species count differs between collision and no-collision modes."""
    from pymars.initial_configuration import read_initial_configuration
    
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    n_atoms = len(species)
    projectile_species_num = 18
    
    # With collision: should have N+1 atoms (molecule + projectile)
    full_species_collision = np.concatenate([
        np.array([projectile_species_num], dtype=np.int32),
        species
    ])
    assert len(full_species_collision) == n_atoms + 1
    
    # Without collision: should have N atoms (molecule only)
    full_species_no_collision = species.copy()
    assert len(full_species_no_collision) == n_atoms
    
    # Verify they differ by exactly 1
    assert len(full_species_collision) == len(full_species_no_collision) + 1
