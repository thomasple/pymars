import pytest
from pathlib import Path
import tempfile


def test_model_file_key_handling():
    """Test that both 'model' and 'model_file' keys are supported"""
    from pymars.md import initialize_collision_simulation
    
    # Create a temporary model file (just for path validation test)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
        temp_model_path = f.name
        f.write("dummy")  # Write something so it's a valid file
    
    try:
        test_data_dir = Path(__file__).parent
        
        # Test with 'model_file' key (as shown in README)
        config_with_model_file = {
            'initial_geometry': str(test_data_dir / "aspirin.xyz"),
            'model_file': temp_model_path,
            'simulation_time': 1.0,
            'dt': 0.001,
            'batch_size': 1,
            'temperature': 300.0,
        }
        
        # This will fail when trying to actually load the model (it's not a real fennol model)
        # but it should get past the key validation
        try:
            system = initialize_collision_simulation(config_with_model_file, verbose=False)
        except Exception as e:
            # We expect this to fail at model loading, not at key validation
            assert "model_file" not in str(e).lower() or "model" not in str(e).lower()
        
        # Test with 'model' key (legacy support)
        config_with_model = {
            'initial_geometry': str(test_data_dir / "aspirin.xyz"),
            'model': temp_model_path,
            'simulation_time': 1.0,
            'dt': 0.001,
            'batch_size': 1,
            'temperature': 300.0,
        }
        
        try:
            system = initialize_collision_simulation(config_with_model, verbose=False)
        except Exception as e:
            # We expect this to fail at model loading, not at key validation
            assert "model_file" not in str(e).lower() or "model" not in str(e).lower()
        
        # Test with neither key - should raise ValueError
        config_without_model = {
            'initial_geometry': str(test_data_dir / "aspirin.xyz"),
            'simulation_time': 1.0,
            'dt': 0.001,
            'batch_size': 1,
            'temperature': 300.0,
        }
        
        with pytest.raises(ValueError, match="Configuration must contain either 'model_file' or 'model' key"):
            system = initialize_collision_simulation(config_without_model, verbose=False)
    
    finally:
        # Clean up temp file
        Path(temp_model_path).unlink(missing_ok=True)
