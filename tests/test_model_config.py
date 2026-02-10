import pytest
from pathlib import Path
import tempfile
import sys


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
            assert "model_file" not in str(e).lower() and "model" not in str(e).lower()
        
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
            assert "model_file" not in str(e).lower() and "model" not in str(e).lower()
        
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


def test_model_directory_added_to_syspath():
    """Test that model directory is added to sys.path for auxiliary file imports"""
    from pymars.md import initialize_collision_simulation
    
    # Create a temporary directory structure with model and auxiliary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a model subdirectory
        model_dir = tmpdir / "model_dir"
        model_dir.mkdir()
        
        # Create a dummy model file
        model_file = model_dir / "test_model.pkl"
        model_file.write_text("dummy")
        
        # Create a dummy auxiliary Python file (like recraster.py)
        aux_file = model_dir / "custom_module.py"
        aux_file.write_text("TEST_VARIABLE = 'model_dir_in_path'")
        
        test_data_dir = Path(__file__).parent
        config = {
            'initial_geometry': str(test_data_dir / "aspirin.xyz"),
            'model_file': str(model_file),
            'simulation_time': 1.0,
            'dt': 0.001,
            'batch_size': 1,
            'temperature': 300.0,
        }
        
        # Store original sys.path
        original_syspath = sys.path.copy()
        model_dir_str = str(model_dir.resolve())
        
        # Ensure model_dir is not in sys.path before the test
        if model_dir_str in sys.path:
            sys.path.remove(model_dir_str)
        
        try:
            # This will fail at model loading, but should add the directory to sys.path first
            try:
                system = initialize_collision_simulation(config, verbose=False)
            except Exception:
                pass  # Expected to fail on actual model loading
            
            # Verify that model directory was added to sys.path
            assert model_dir_str in sys.path, f"Model directory {model_dir_str} should be in sys.path"
            
            # Verify we can import the auxiliary file
            import custom_module
            assert custom_module.TEST_VARIABLE == 'model_dir_in_path'
            
        finally:
            # Clean up sys.modules and sys.path
            if 'custom_module' in sys.modules:
                del sys.modules['custom_module']
            if model_dir_str in sys.path:
                sys.path.remove(model_dir_str)

