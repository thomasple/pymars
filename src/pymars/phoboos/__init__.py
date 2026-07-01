import argparse
import yaml
import os
import sys
#import numpy as np
from pymars.__init__ import _Tee

def main():

    parser = argparse.ArgumentParser(prog="phoboos",
        description="phoboos: A FENNIX-powered molecular geometry package"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    opt_parser = subparsers.add_parser("opt")
    opt_parser.add_argument("input_file", type=str, help="Path to the input configuration file"
    )

    spt_parser = subparsers.add_parser("spt")
    spt_parser.add_argument("input_file", type=str, help="Path to the input configuration file"
    )

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        simulation_parameters = yaml.safe_load(f)

    # --- Set up internal logging to <prefix>.out as early as possible, so that
    # essentially all terminal output (starting with the very first print below)
    # is also captured to disk, without requiring the user to redirect manually.
    # <prefix> is derived from input_parameters.initial_geometry (e.g. "aspirin.xyz"
    # -> "aspirin.out").
    input_yaml_dir = os.path.dirname(os.path.abspath(args.input_file))
    _early_input_params = simulation_parameters.get("input_parameters", {})
    _early_initial_xyz = _early_input_params.get("initial_geometry", None)
    if _early_initial_xyz:
        _log_prefix = os.path.splitext(os.path.basename(_early_initial_xyz))[0]
    else:
        raise ValueError("Missing 'initial_geometry' in input_parameters of the input YAML file.")
    log_file_path = os.path.join(input_yaml_dir, f"{_log_prefix}.out")
    _log_fh = open(log_file_path, "a")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print("\n") #Header
    print("################################################################")
    print("#  PhoBOOS: PhtHOn BOOSter for molecular geometry calculations #")
    print("#       A FENNIX-powered molecular geometry package            #")
    print("################################################################")
    print(f"# Writing output to: {log_file_path}")

    # Print execution folder (working directory where the command is run, which may differ from installation path).
    print(f"# Running from folder: {os.getcwd()}")

    # Set FENNOL_MODULES_PATH BEFORE any fennol imports
    # This must be done before importing anything that imports fennol
    calc_params = simulation_parameters.get("calculation_parameters", simulation_parameters)
    model_file = calc_params.get("model", calc_params.get("model_file", None))
    if model_file:
        from pathlib import Path
        model_path = Path(model_file).resolve()
        if model_path.exists():
            model_dir = str(model_path.parent)
            os.environ['FENNOL_MODULES_PATH'] = model_dir
    if not model_file:
        raise ValueError("Missing 'model' in calculation_parameters of the input YAML file.")

    # Note: postpone importing fennol/.utils (which may import jax) until after
    # we've set CUDA_VISIBLE_DEVICES and configured JAX so device detection
    # happens with the intended environment. See below where these imports
    # are performed after importing jax.

   # Set the device
    if "MARS_DEVICE" in os.environ:
        device = os.environ["MARS_DEVICE"].lower()
        print(f"# Setting device from env MARS_DEVICE={device}")
    else:
        # prefer calculation_parameters.device if present
        device = calc_params.get("device", simulation_parameters.get("device", "cpu")).lower()
        """@keyword[fennol_md] device
        Computation device. Options: 'cpu', 'cuda:N', 'gpu:N' where N is device number.
        Default: 'cpu'
        """
    
        # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing jax
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No GPUs
    elif device.startswith("cuda") or device.startswith("gpu"):
        # Extract GPU index (e.g. "cuda:1" -> "1")
        if ":" in device:
            gpu_index = device.split(":", 1)[1]
        else:
            gpu_index = "0"
        
        # Make ONLY this physical GPU visible to the entire process
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        print(f"# Set CUDA_VISIBLE_DEVICES={gpu_index}")
        
        simulation_parameters["torch_device"] = "cuda:0"  # Always cuda:0 within visible set
        device = "gpu"

    # Now import jax (AFTER environment is set)
    import jax
    import jax.numpy as jnp

    # Configure JAX (always use first/only visible device)
    if device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    elif device == "gpu":
        jax.config.update("jax_platforms", "")
        jax.config.update("jax_cuda_visible_devices", "0")  # First visible GPU

    # Select the first device (should be the one exposed by CUDA_VISIBLE_DEVICES)
    try:
        _device = jax.devices()[0]
    except RuntimeError as e:
        print(f"# Error initializing JAX device: {e}")
        print("# Exiting due to device initialization failure.")
        print("# Detailed error information:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    jax.config.update("jax_default_device", _device)
    jax.config.update("jax_default_matmul_precision", "highest")
 
    # Debug prints to help diagnose device selection issues
    #print(f"# CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    #try:
    #    print(f"# jax.devices(): {jax.devices()}")
    #except Exception as e:
    #    print(f"# Error listing jax.devices(): {e}")
    # Print a concise message showing which device will be used for computations
    used_dev = simulation_parameters.get("torch_device", None)
    if used_dev is None:
        # fallback to JAX's detected device
        used_dev = str(_device)
    print(f"# Using device: {used_dev}")

    # Now it's safe to import fennol utilities and convert units because
    # JAX has been imported and configured with the intended device.
    from fennol.utils.input_parser import convert_dict_units
    from pymars.utils import us

    # first take common inputs from both singlepoint and optimization:
    input_params = simulation_parameters.get("input_parameters", {})
    initial_xyz = input_params.get("initial_geometry", None)
    if not initial_xyz:
        raise ValueError("Missing 'initial_geometry' in input_parameters of the input YAML file.")
    total_charge = input_params.get("total_charge", 0)
    with open(initial_xyz, "r", encoding="utf-8") as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    #Input section printout
    print("\n")
    print("##################################################")
    print("#                   INPUT SECTION                 ")
    print("##################################################")
    print(f"#  Model used: {os.path.basename(model_file)}")
    if args.command == "spt":
        print("#  Single-point energy calculation")
    elif args.command == "opt":
        print("#  Geometry optimization")
    print(f"#  Initial geometry: {initial_xyz}")
    print(f"#  Number of atoms: {natoms}    Charge: {total_charge}")
    # Skip atom count (line 0) and comment (line 1)
    for line in lines[2:]:
        print(line.rstrip())
    print("##################################################")
    print("#               END OF INPUT SECTION             ")
    print("##################################################")

    
    #from here on, we can branch into the specific commands (opt or spt) and handle their respective parameters and execution.

    #for the singlepoint there are no additional parameters, so we can directly call the singlepoint function if the command is "spt"
    if args.command == "spt":
        from .singlepoint import run_spt
        print("Performing single-point energy calculation...")
        run_spt(initial_xyz, model_file=model_file, total_charge=total_charge)
        print ("\nPhoBOOS terminated normally.")
        sys.exit(0)
    
    elif args.command == "opt":
        #for the optimization command, we need to handle additional parameters specific to optimization
        #double_precision = simulation_parameters.get("calculation_parameters", {}).get("double_precision", True)
        tolerance = float(simulation_parameters.get("optimization_parameters", {}).get("tolerance", 1e-2))
        dx_max = simulation_parameters.get("optimization_parameters", {}).get("dx_max", 0.2)
        dt_dyn = simulation_parameters.get("optimization_parameters", {}).get("dt_dyn", 2.0)
        dt_ps = dt_dyn * 1e-3  # convert fs to ps
        max_steps = simulation_parameters.get("optimization_parameters", {}).get("max_steps", 10000)
        save_steps = simulation_parameters.get("optimization_parameters", {}).get("save_steps", -1)
        from .optiminimize import run_opt
        print("Performing geometry optimization...")
        run_opt(xyz_file=initial_xyz,
            model_file=model_file,
            dt=dt_ps,
            total_charge=total_charge,
            tolerance=tolerance,
            max_steps=max_steps,
            keep_every=save_steps,
            dxmax=dx_max,
            )

        print ("\nPhoBOOS terminated normally.")
        sys.exit(0)
    else:
        print(f"Unknown command: {args.command}")
        print("PhoBOOS terminated abnormally")
        sys.exit(1)
