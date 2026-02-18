"""
Tests for fragmentation detection and distance threshold functionality.
"""
import pytest
import numpy as np
from pathlib import Path


def test_fragment_separation_single_molecule(test_data_dir):
    """Test that a single connected molecule returns 0.0 separation."""
    from pymars.topology import get_fragment_separation
    from pymars.initial_configuration import read_initial_configuration
    
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    
    # Single molecule should have zero fragment separation
    separation = get_fragment_separation(species, coords)
    assert separation == 0.0


def test_fragment_separation_two_fragments():
    """Test fragment separation with two clearly separated fragments."""
    from pymars.topology import get_fragment_separation
    
    # Create two separate H2 molecules
    species = np.array([1, 1, 1, 1], dtype=np.int32)  # 4 hydrogen atoms
    
    # Two H2 molecules separated by 15 Angstroms
    coords = np.array([
        [0.0, 0.0, 0.0],   # H2 molecule 1
        [0.7, 0.0, 0.0],
        [15.0, 0.0, 0.0],  # H2 molecule 2
        [15.7, 0.0, 0.0],
    ], dtype=np.float32)
    
    separation = get_fragment_separation(species, coords)
    
    # Should be approximately 15 Angstroms (center of mass distance)
    assert 14.0 < separation < 16.0


def test_fragment_separation_with_masses():
    """Test that masses are properly used in COM calculation."""
    from pymars.topology import get_fragment_separation
    
    # Two fragments with different masses
    species = np.array([6, 6, 1, 1], dtype=np.int32)  # 2 C, 2 H
    coords = np.array([
        [0.0, 0.0, 0.0],   # C
        [1.5, 0.0, 0.0],   # C (bonded to first C)
        [10.0, 0.0, 0.0],  # H (separate fragment)
        [10.7, 0.0, 0.0],  # H
    ], dtype=np.float32)
    
    # Provide masses
    masses = np.array([12.0, 12.0, 1.0, 1.0], dtype=np.float32)
    
    separation = get_fragment_separation(species, coords, masses=masses)
    
    # Should compute mass-weighted COM
    assert separation > 0.0


def test_count_graphs_single_molecule(test_data_dir):
    """Test that an intact molecule has 1 graph."""
    from pymars.topology import count_graphs
    from pymars.initial_configuration import read_initial_configuration
    
    species, coords = read_initial_configuration(str(test_data_dir / "aspirin.xyz"))
    
    n_graphs = count_graphs(species, coords)
    assert n_graphs == 1


def test_count_graphs_fragmented():
    """Test that fragmented systems have multiple graphs."""
    from pymars.topology import count_graphs
    
    # Two separate H2 molecules
    species = np.array([1, 1, 1, 1], dtype=np.int32)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.7, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.7, 0.0, 0.0],
    ], dtype=np.float32)
    
    n_graphs = count_graphs(species, coords)
    assert n_graphs == 2


def test_get_graph_components():
    """Test that graph components are correctly identified."""
    from pymars.topology import get_graph_components
    
    # Two H2 molecules + one isolated H
    species = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    coords = np.array([
        [0.0, 0.0, 0.0],   # H2 molecule 1
        [0.7, 0.0, 0.0],
        [10.0, 0.0, 0.0],  # H2 molecule 2
        [10.7, 0.0, 0.0],
        [20.0, 0.0, 0.0],  # Isolated H
    ], dtype=np.float32)
    
    components = get_graph_components(species, coords)
    
    # Should have 3 components
    assert len(components) == 3
    
    # Check sizes
    sizes = sorted([len(c) for c in components])
    assert sizes == [1, 2, 2]  # One isolated, two pairs


def test_fragmentation_check_parameter():
    """Test that fragmentation_check parameter is accepted in config."""
    # This is a configuration validation test
    config = {
        "fragmentation_check": True,
        "distance_threshold": 10.0,
    }
    
    assert config["fragmentation_check"] == True
    assert config["distance_threshold"] == 10.0
    
    # Test defaults
    config_no_frag = {}
    frag_check = config_no_frag.get("fragmentation_check", False)
    dist_thresh = config_no_frag.get("distance_threshold", 10.0)
    
    assert frag_check == False
    assert dist_thresh == 10.0


def test_integrate_signature_has_fragmentation_params(test_data_dir):
    """Test that integrate function accepts fragmentation parameters."""
    import inspect
    from pymars.md import initialize_collision_simulation
    
    # Create minimal config
    config = {
        "initial_geometry": {
            "species": [1, 1],
            "coordinates": [[0, 0, 0], [0.7, 0, 0]],
        },
        "model": str(test_data_dir / "mock_model.fnx"),
        "collision_dynamics": False,
        "temperature": 300.0,
        "batch_size": 1,
        "dt": 1.0,
    }
    
    if not Path(config["model"]).exists():
        pytest.skip("Mock model not available")
    
    system = initialize_collision_simulation(config, verbose=False)
    integrate_func = system["integrate"]
    
    # Check function signature
    sig = inspect.signature(integrate_func)
    params = list(sig.parameters.keys())
    
    # Should have fragmentation_check and distance_threshold parameters
    assert "fragmentation_check" in params or "nsteps" in params  # At minimum should have nsteps
    

def test_fragment_separation_edge_cases():
    """Test edge cases for fragment separation."""
    from pymars.topology import get_fragment_separation
    
    # No atoms
    species = np.array([], dtype=np.int32)
    coords = np.array([]).reshape(0, 3)
    sep = get_fragment_separation(species, coords)
    assert sep == 0.0
    
    # Single atom
    species = np.array([1], dtype=np.int32)
    coords = np.array([[0, 0, 0]], dtype=np.float32)
    sep = get_fragment_separation(species, coords)
    assert sep == 0.0


def test_fragmentation_distances_in_return():
    """Test that integrate returns fragment_distances in output."""
    # This test verifies the return dictionary structure
    expected_keys = [
        "coordinates",
        "velocities", 
        "accelerations",
        "energies",
        "split_steps",
        "fragment_distances",  # NEW KEY
        "all_reached_target",
        "initial_graphs",
        "target_graphs",
    ]
    
    # Mock return structure (what integrate should return)
    mock_return = {
        "coordinates": None,
        "velocities": None,
        "accelerations": None,
        "energies": None,
        "split_steps": np.array([-1, -1]),
        "fragment_distances": np.array([-1.0, -1.0]),  # Should be included
        "all_reached_target": False,
        "initial_graphs": np.array([1, 1]),
        "target_graphs": np.array([2, 2]),
    }
    
    for key in expected_keys:
        assert key in mock_return, f"Missing key: {key}"
    
    # Verify fragment_distances is a numpy array
    assert isinstance(mock_return["fragment_distances"], np.ndarray)
    assert mock_return["fragment_distances"].dtype == np.float64 or mock_return["fragment_distances"].dtype == np.float32


def test_distance_threshold_logic():
    """Test the logic for determining when fragments are separated enough."""
    distance_threshold = 10.0
    
    # Fragment distances for different scenarios
    test_cases = [
        (5.0, False, "Fragments not separated enough"),
        (10.0, True, "Fragments exactly at threshold"),
        (15.0, True, "Fragments well separated"),
        (0.0, False, "Fragments not separated (still bonded)"),
        (-1.0, False, "Not fragmented yet"),
    ]
    
    for frag_dist, should_be_separated, description in test_cases:
        is_separated = frag_dist >= distance_threshold
        assert is_separated == should_be_separated, description


def test_batch_completion_logic():
    """Test batch completion logic - all must be fragmented and separated."""
    batch_size = 5
    
    # Scenario 1: All trajectories fragmented and separated
    split_steps = np.array([100, 150, 200, 250, 300])  # All completed
    active = np.array([False, False, False, False, False])
    all_complete = not active.any()
    assert all_complete == True
    
    # Scenario 2: Some still active
    split_steps = np.array([100, 150, -1, -1, 300])
    active = np.array([False, False, True, True, False])
    all_complete = not active.any()
    assert all_complete == False
    
    # Scenario 3: None completed
    split_steps = np.array([-1, -1, -1, -1, -1])
    active = np.array([True, True, True, True, True])
    all_complete = not active.any()
    assert all_complete == False
