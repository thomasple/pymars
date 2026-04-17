import argparse
import yaml
import numpy as np
import os
import time
import re
import shutil

__all__ = []

def main() -> None:
    # Print installed package path (directory containing this __init__.py module).
    print(f"# Installation path: {os.path.dirname(os.path.abspath(__file__))}")
    # Print execution folder (working directory where the command is run, which may differ from installation path).
    print(f"# Running from folder: {os.getcwd()}")

    parser = argparse.ArgumentParser(
        description="pymars: A molecular collision simulation package"
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input configuration file"
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        simulation_parameters = yaml.safe_load(f)
    
    # Set FENNOL_MODULES_PATH BEFORE any fennol imports
    # This must be done before importing utils (which imports fennol) and md (which imports fennol)
    calc_params = simulation_parameters.get("calculation_parameters", simulation_parameters)
    model_file = calc_params.get("model", calc_params.get("model_file", None))
    if model_file:
        from pathlib import Path
        model_path = Path(model_file).resolve()
        if model_path.exists():
            model_dir = str(model_path.parent)
            os.environ['FENNOL_MODULES_PATH'] = model_dir

    # --- Restart logic: determine restart file name ---
    input_params = simulation_parameters.get("input_parameters", {})
    initial_xyz = input_params.get("initial_geometry", None)
    if initial_xyz is not None:
        base_xyz = os.path.basename(initial_xyz)
        restart_file_name = os.path.splitext(base_xyz)[0] + ".dyn.restart"
    else:
        restart_file_name = "traj.dyn.restart"
    restart_traj = bool(calc_params.get("restart_traj", False))

    # Determine restart file path: place the restart file next to the input YAML
    input_yaml_dir = os.path.dirname(os.path.abspath(args.input_file))
    restart_file = os.path.join(input_yaml_dir, restart_file_name)
    # Also show where we'll look/save restart files
    # (useful when working directories or folder names change)
    print(f"# Restart file will be: {restart_file_name}")

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
        import sys
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
    from .utils import us
    simulation_parameters = convert_dict_units(simulation_parameters, us)
    
    ### Set per-trajectory seeds for reproducibility / stochastic initialization
    general_params = simulation_parameters.get("general_parameters", simulation_parameters)
    batch_size_cfg = int(general_params.get("batch_size", 1))
    if batch_size_cfg < 1:
        raise ValueError(f"general_parameters.batch_size must be >= 1, got {batch_size_cfg}")

    seed_cfg = general_params.get("seed", None)
    _seed_rng = np.random.default_rng()
    user_provided_seed_list = False

    # Keep user-facing input key as `seed`.
    # Accepted forms:
    # - batch_size > 1: list/tuple/ndarray of seeds OR comma-separated string "1,2,3"
    # - batch_size == 1: scalar seed value (int-like) is used directly
    # Fallback behavior:
    # - batch_size > 1 and seed missing/scalar/non-list -> random per-trajectory seeds
    # - batch_size == 1 and seed missing -> one random seed
    if isinstance(seed_cfg, str) and "," in seed_cfg and batch_size_cfg > 1:
        parsed = [p.strip() for p in seed_cfg.split(",") if p.strip() != ""]
        trajectory_seeds = [int(p) for p in parsed]
        user_provided_seed_list = True
    elif isinstance(seed_cfg, (list, tuple, np.ndarray)):
        trajectory_seeds = [int(s) for s in seed_cfg]
        user_provided_seed_list = True
    elif batch_size_cfg == 1 and seed_cfg is not None:
        # Single-trajectory run: keep scalar seed exactly as provided.
        trajectory_seeds = [int(seed_cfg)]
    else:
        trajectory_seeds = [
            int(x) for x in _seed_rng.integers(0, 2**32 - 1, size=batch_size_cfg, dtype=np.uint32)
        ]

    if len(trajectory_seeds) != batch_size_cfg:
        raise ValueError(
            f"general_parameters.seed provides {len(trajectory_seeds)} value(s), "
            f"but batch_size is {batch_size_cfg}. Provide one seed per trajectory."
        )

    # For batch runs, enforce unique seeds across trajectories.
    # - user-provided lists with duplicates: fail fast with clear message
    # - randomly generated seeds with duplicates: replace duplicates with new unique seeds
    if batch_size_cfg > 1:
        if user_provided_seed_list and len(set(trajectory_seeds)) != len(trajectory_seeds):
            raise ValueError("same seed repeated, ensure all seeds are unique")

        if not user_provided_seed_list:
            seen = set()
            for i, s in enumerate(trajectory_seeds):
                while s in seen:
                    s = int(_seed_rng.integers(0, 2**32 - 1, dtype=np.uint32))
                trajectory_seeds[i] = s
                seen.add(s)

    print(f"# Per-trajectory seeds:")
    print("# " + " / ".join(f"traj[{i}]_seed={s}" for i, s in enumerate(trajectory_seeds)))

    # Pass resolved per-trajectory seeds to md initialization code.
    if "general_parameters" in simulation_parameters and isinstance(simulation_parameters["general_parameters"], dict):
        simulation_parameters["general_parameters"]["seed_list"] = trajectory_seeds
    else:
        simulation_parameters["seed_list"] = trajectory_seeds
    
    ### Set precision (double precision / float64)
    calc_params = simulation_parameters.get("calculation_parameters", simulation_parameters)
    enable_x64 = calc_params.get("double_precision", False)
    jax.config.update("jax_enable_x64", enable_x64)
    if enable_x64:
        print("# Using double precision (float64)")
    else:
        print("# Using single precision (float32)")

    # Import md module AFTER setting FENNOL_MODULES_PATH
    from .md import initialize_collision_simulation
    system = initialize_collision_simulation(simulation_parameters)

    integrate = system["integrate"]
    # Default: start from initial configuration
    coordinates = system["coordinates"]
    velocities = system["velocities"]
    accelerations = system["accelerations"]
    species = system["species"]
    masses = system["masses"]
    batch_size = system["batch_size"]
    start_step = 0

    # If restart_traj is true, try to locate a restart file. numpy.savez
    # appends a .npz extension if it is not present, so check both
    # '<prefix>.dyn.restart' and '<prefix>.dyn.restart.npz'.
    if restart_traj:
        candidates = [restart_file, restart_file + ".npz"]
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        if found is not None:
            print(f"# Restarting trajectory from {found}")
            arr = np.load(found)
            coordinates = arr["coordinates"]
            velocities = arr["velocities"]
            accelerations = arr["accelerations"]
            if "step" in arr:
                start_step = int(arr["step"])
            else:
                start_step = 0
        else:
            print(f"# restart_traj is True but no restart file found among: {candidates}; starting from initial configuration.")
    
    # Convert atomic numbers to element symbols for trajectory output
    from ase.data import chemical_symbols
    element_symbols = [chemical_symbols[z] for z in species]

    # Extract nested parameters or use legacy flat format
    general_params = simulation_parameters.get("general_parameters", simulation_parameters)
    dynamic_params = simulation_parameters.get("dynamic_parameters", simulation_parameters)
    output_params = simulation_parameters.get("output_details", simulation_parameters)
    
    # Get simulation time parameters
    dt = system["dt"]  # ps
    dt_fs = dt * 1000.0  # Convert dt from ps to fs
    if "step_dyn" in dynamic_params:
        n_steps = int(dynamic_params["step_dyn"])
        simulation_time = n_steps * dt
    elif "simulation_time" in simulation_parameters:
        simulation_time = simulation_parameters["simulation_time"]  # ps
        n_steps = round(simulation_time / dt)
        simulation_time = n_steps * dt  # adjust to exact number of steps
    else:
        raise ValueError("Either step_dyn or simulation_time must be specified")
    
    print(f"# Running simulation for {simulation_time} ps ({n_steps} steps of {dt_fs} fs)")

    # Get output parameters
    save_steps = general_params.get("save_steps", general_params.get("print_step", 100))
    save_energy = general_params.get("save_energy", general_params.get("energy_steps", 100))
    
    traj_file = str(output_params.get("trajectory_file", output_params.get("traj_file", "trajectory.xyz")))
    traj_paths = []
    write_traj = traj_file.lower() != "none"
    if write_traj:
        assert traj_file.endswith(".xyz"), "Only .xyz output format is supported currently."
        from fennol.utils.io import write_xyz_frame
        if batch_size ==1:
            traj_paths = [traj_file]
            ftraj = [open(traj_paths[0], "w")]
        else:
            num=len(str(batch_size-1))
            traj_paths = [traj_file.replace(".xyz", f'.{i:0{num}}.xyz') for i in range(batch_size)]
            ftraj = [open(p, "w") for p in traj_paths]

    def _indexed_paths(base_path, n):
        """Return n unique per-trajectory paths by injecting _i before extension.

        Uses os.path.splitext so naming works even when ".out" is missing or appears
        in the middle of the filename.
        Examples:
          - energies.out -> energies_0.out, energies_1.out
          - energies     -> energies_0, energies_1
          - my.out.file  -> my.out_0.file, my.out_1.file
        """
        root, ext = os.path.splitext(base_path)
        return [f"{root}_{i}{ext}" for i in range(n)]

    # Get energy output file(s)
    # For batch_size > 1 and a single configured filename, generate per-trajectory
    # unique files by injecting _i before the extension via splitext.
    energy_file = output_params.get("energies_file", output_params.get("energy_output_file", None))
    energy_files = []
    if energy_file:
        if isinstance(energy_file, str):
            if batch_size == 1:
                energy_files = [energy_file]
            else:
                energy_files = _indexed_paths(energy_file, batch_size)
        elif isinstance(energy_file, (list, tuple)):
            # Explicit per-trajectory files provided by user.
            energy_files = list(energy_file)
        else:
            raise TypeError(
                "output_details.energies_file must be either a string or a list/tuple of strings"
            )

        # Validate list length early to avoid runtime IndexError in md.write_energy_output
        # (which indexes output_file[b] for b in range(batch_size)).
        if len(energy_files) != batch_size:
            raise ValueError(
                f"output_details.energies_file defines {len(energy_files)} file(s) "
                f"but batch_size is {batch_size}. Provide exactly one energy file per trajectory."
            )
            

    # Summary file setup: one summary file per trajectory in batch mode.
    # Uses the same splitext-safe indexing logic as energy files.
    summary_cfg = output_params.get("summary_file", None)
    summary_files = None
    if summary_cfg:
        if isinstance(summary_cfg, str):
            if batch_size == 1:
                # Single trajectory: use the configured summary filename as-is
                summary_files = [summary_cfg]
            else:
                # Multiple trajectories: append _i suffix to create separate summary per trajectory
                summary_files = _indexed_paths(summary_cfg, batch_size)
        elif isinstance(summary_cfg, (list, tuple)):
            # User provided explicit list of summary filenames (one per trajectory)
            summary_files = list(summary_cfg)
        else:
            raise TypeError(
                "output_details.summary_file must be either a string or a list/tuple of strings"
            )

        # Validate list length early to avoid downstream indexing mismatches.
        if len(summary_files) != batch_size:
            raise ValueError(
                f"output_details.summary_file defines {len(summary_files)} file(s) "
                f"but batch_size is {batch_size}. Provide exactly one summary file per trajectory."
            )

    # Track variance option
    track_variance = bool(output_params.get("track_variance", False))

    @jax.jit
    def mean_energies_and_remove_com(coordinates, velocities,epots):
        com = jnp.mean(coordinates[:,1:,:], axis=1, keepdims=True)
        coordinates = coordinates - com  # center at COM
        ekin = 0.5 * jnp.sum(
            masses[None,:] * jnp.sum(velocities**2, axis=-1)
        ) / batch_size  # kcal/mol
        epot = jnp.mean(epots)
        return coordinates, ekin+epot, epot, ekin
        

    time_start = time.time()
    time0 = time_start
    
    # Determine time format precision based on dt
    if dt_fs >= 1.0:
        time_decimals = 0
    else:
        time_decimals = len(str(dt_fs).split('.')[-1].rstrip('0'))
    
    # Get save_summary parameter
    save_summary = general_params.get("save_summary", None)
    if save_summary is not None and save_summary < save_energy:
        raise ValueError(f"save_summary ({save_summary}) must be >= save_energy ({save_energy})")
    
    # Get thermostat parameters
    thermostat_params = simulation_parameters.get("thermostat_parameters", simulation_parameters)
    is_nve = thermostat_params.get("NVE_thermostat", True)  # Default to NVE
    
    # Prepare summary file(s) if save_summary parameter is set (interval between summary writes in steps)
    # If save_summary is None, summary output is disabled
    if save_summary is None:
        # No summary requested; disable summary output entirely
        summary_files = None
    elif summary_files is None:
        # Fallback: if save_summary is set but no summary_file config provided, create defaults
        if batch_size == 1:
            summary_files = ["summary.out"]
        else:
            # Default naming for per-trajectory summaries when batch_size > 1
            summary_files = [f"summary_{i}.out" for i in range(batch_size)]
    
    # Energy tracking for summary and drift calculation
    # These are initialized per-trajectory for batch mode (arrays with shape batch_size)
    initial_total_energy = None  # Baseline total energy per trajectory (set on first energy_data arrival)
    max_energy_drift = None      # Max energy deviation from baseline per trajectory (1D array in batch mode)
    energy_history = []          # Accumulate energy frames from integrator for summary statistics (cleared every save_summary steps)
    
    header = f"#{'Step':>10} {'Time[fs]':>12} {'Etot':>12} {'Epot':>12} {'Ekin':>12} {'ns/day':>12}"
    if batch_size == 1:
        print(header)

    for istep in range(start_step, n_steps):
        coordinates, velocities, accelerations, energies, energy_data, frame_variance = integrate(
            coordinates, velocities, accelerations,
            step=istep,
            energy_output_file=energy_files if energy_file else None,
            energy_steps=save_energy
        )

        # Collect energy data for summary statistics
        if energy_data is not None:
            energy_history.append(energy_data)
            # Set initial energies for drift calculation (per-trajectory)
            current_total = np.atleast_1d(np.asarray(energy_data['total_energies'], dtype=float))
            if initial_total_energy is None:
                initial_total_energy = current_total.copy()
                max_energy_drift = np.zeros_like(current_total, dtype=float)
            # Track maximum energy drift per trajectory
            drift = np.abs(current_total - initial_total_energy)
            max_energy_drift = np.maximum(max_energy_drift, drift)

        if (istep + 1) % save_steps == 0:
            time_elapsed = time.time() - time0
            time0 = time.time()
            time_per_step = time_elapsed / save_steps
            step_per_day = 60*60*24 / time_per_step
            ns_per_step = dt*us.NS
            ns_per_day = ns_per_step * step_per_day

            coordinates, total_energy, potential_energy, ekin = mean_energies_and_remove_com(
                coordinates, velocities,energies
            )

            # Format time in femtoseconds with appropriate precision
            time_fs = (istep+1)*dt * 1000.0  # Convert ps to fs
            if time_decimals == 0:
                time_str = f"{time_fs:.0f}"
            else:
                time_str = f"{time_fs:.{time_decimals}f}"

            line = f" {istep+1:10} {time_str:>12} {total_energy:12.3f} {potential_energy:12.3f} {ekin:12.3f} {ns_per_day:12.1f}"
            if batch_size == 1:
                print(line)

            if write_traj:
                coords = np.array(coordinates)
                for i in range(batch_size):
                    write_xyz_frame(
                        ftraj[i],
                        element_symbols,
                        coords[i],
                        comment=f"Step {istep+1} E_pot={potential_energy:.6f}",
                    )
        
        # Write summary output if requested (every save_summary steps)
        # Summary aggregates energy statistics for each trajectory independently over the interval
        if save_summary is not None and summary_files is not None and (istep + 1) % save_summary == 0:
            from fennol.utils.io import human_time_duration
            
            # Calculate per-trajectory statistics over the summary interval (accumulated in energy_history)
            n_summary_steps = len(energy_history)
            if n_summary_steps > 0:
                # Helper function: extract per-trajectory value from energy dict
                # Handles case where energy data is stored as batch array (batch_size,) or per-traj
                def _extract_traj_val(entry, key, b):
                    arr = np.atleast_1d(np.asarray(entry[key], dtype=float))
                    # Index b selects the b-th trajectory; fallback to last if out of bounds
                    idx = b if b < arr.size else arr.size - 1
                    return arr[idx]
                
                # Time statistics
                time_for_interval = time.time() - time0
                simulated_time_ps = (istep + 1) * dt
                total_simulated_time_ps = simulated_time_ps * batch_size
                total_elapsed_time = time.time() - time_start
                
                # Average calculation speed
                avg_ns_per_day = ns_per_day  # Use the last calculated value
                avg_step_per_s = save_summary / time_for_interval
                
                # Estimate remaining time
                remaining_steps = n_steps - (istep + 1)
                est_remaining_time = remaining_steps * (total_elapsed_time / (istep + 1))
                est_total_duration = total_elapsed_time + est_remaining_time
                
                # Write summary for each trajectory independently (per-trajectory stats to per-trajectory file)
                for b in range(batch_size):
                    if b >= len(summary_files):
                        continue

                    # Extract trajectory b's energy series from accumulated history (all frames since last summary)
                    all_etot = np.array([_extract_traj_val(e, 'total_energies', b) for e in energy_history], dtype=float)
                    all_epot = np.array([_extract_traj_val(e, 'potential_energies', b) for e in energy_history], dtype=float)
                    all_ekin = np.array([_extract_traj_val(e, 'kinetic_energies', b) for e in energy_history], dtype=float)
                    all_temp = np.array([_extract_traj_val(e, 'temperatures', b) for e in energy_history], dtype=float)

                    # Calculate mean and standard deviation for trajectory b over the summary interval
                    avg_etot = np.mean(all_etot)
                    std_etot = np.std(all_etot)
                    avg_epot = np.mean(all_epot)
                    std_epot = np.std(all_epot)
                    avg_ekin = np.mean(all_ekin)
                    std_ekin = np.std(all_ekin)
                    avg_temp = np.mean(all_temp)
                    std_temp = np.std(all_temp)

                    # Compute energy drift percentage for trajectory b (NVE energy conservation quality metric)
                    drift_percent = 0.0
                    if initial_total_energy is not None and b < len(initial_total_energy) and initial_total_energy[b] != 0:
                        drift_percent = (max_energy_drift[b] / abs(initial_total_energy[b])) * 100.0

                    # Write trajectory-specific summary to its dedicated summary file
                    with open(summary_files[b], 'a') as f:
                        f.write("##################################################\n")
                        f.write(f"# Step {istep+1:_} of {n_steps:_}  ({((istep+1)/n_steps*100):.3f} %)\n")
                        f.write(f"# Simulated time      : {simulated_time_ps:.3f} ps\n")
                        f.write(f"# Tot. Simu. time     : {total_simulated_time_ps:.3f} ps\n")
                        f.write(f"# Tot. elapsed time   : {human_time_duration(total_elapsed_time)}\n")
                        f.write(f"# Avg. calc. speed: {avg_ns_per_day:.2f} ns/day  ( {avg_step_per_s:.2f} step/s )\n")
                        f.write(f"# Est. total duration   : {human_time_duration(est_total_duration)}\n")
                        f.write(f"# Est. remaining time : {human_time_duration(est_remaining_time)}\n")
                        f.write(f"# Time for {save_summary:_} steps : {human_time_duration(time_for_interval)}\n")
                        # Only show energy drift for NVE simulations
                        if is_nve:
                            f.write(f"# Maximum energy drift (NVE): {max_energy_drift[b]:.2f} kcal/mol ({drift_percent:.1f}%)\n")
                        f.write(f"# Averages over last {n_summary_steps * save_energy:_} steps :\n")
                        f.write(f"#   Etot       : {avg_etot:11.2f}   +/- {std_etot:9.3f}  kcal/mol\n")
                        f.write(f"#   Epot       : {avg_epot:11.2f}   +/- {std_epot:9.2f}  kcal/mol\n")
                        f.write(f"#   Ekin       : {avg_ekin:11.3f}   +/- {std_ekin:9.2f}  kcal/mol\n")
                        f.write(f"#   Temper     : {avg_temp:11.1f}   +/- {std_temp:9.0f}  Kelvin\n")
                        f.write("##################################################\n\n")
                
                # Clear energy history for next interval
                energy_history.clear()


    if write_traj:
        for f in ftraj: 
            f.close()
    # --- Always save last state for restart ---
    # Ensure directory exists (should, as it's input yaml dir)
    try:
        os.makedirs(os.path.dirname(restart_file), exist_ok=True)
    except Exception:
        pass
    # Convert JAX arrays to numpy before saving to ensure compatibility
    try:
        save_coords = np.asarray(coordinates)
        save_vels = np.asarray(velocities)
        save_accs = np.asarray(accelerations)
    except Exception:
        save_coords = coordinates
        save_vels = velocities
        save_accs = accelerations

    # Ensure we know the exact filename np.savez will create
    restart_save_file = restart_file if restart_file.endswith(".npz") else restart_file + ".npz"
    np.savez(restart_save_file,
        coordinates=save_coords,
        velocities=save_vels,
        accelerations=save_accs,
        step=n_steps
    )
    print(f"# Saved last state to {restart_save_file}")
    from fennol.utils.io import human_time_duration
    total_time = time.time() - time_start
    nsperday = (simulation_time / total_time)*60*60*24*us.NS
    print(f"# {simulation_time*us.PS} ps simulation completed in {human_time_duration(total_time)} ({nsperday:.1f} ns/day)")

    # ================================================================ #
    # Batch artifact export: move per-trajectory outputs into SIMXXXXX dirs
    # ================================================================ #
    # This section organizes batch run outputs into timestamped simulation folders (SIM00000, SIM00001, ...)
    # Each SIM folder contains a single trajectory's outputs (trajectory.xyz, energies.out, summary.out)
    # plus copies of the input configuration (input.yaml, starting geometry)
    # This structure enables easy tracking and comparison of multiple parallel simulations
    
    #print(f"# [BATCH_DEBUG] Entering batch artifact export phase (batch_size={batch_size})")
    if batch_size <= 1:
        #print("# [BATCH_DEBUG] batch_size <= 1, skipping SIM export logic")
        return

    def _resolve_existing_path(path_str, bases):
        """Resolve a file path by checking absolute path, searching candidate bases, or current dir."""
        if path_str is None:
            return None
        if os.path.isabs(path_str) and os.path.exists(path_str):
            return path_str
        for b in bases:
            cand = os.path.join(b, path_str)
            if os.path.exists(cand):
                return cand
        if os.path.exists(path_str):
            return os.path.abspath(path_str)
        return None

    def _move_if_exists(src_path, dst_dir, dst_name=None):
        """Move a file from src_path to dst_dir, optionally renaming to dst_name. Overwrites existing target."""
        if src_path is None:
            #print(f"# [BATCH_DEBUG] move skipped (source is None) -> dst_dir={dst_dir}")
            return
        if os.path.isfile(src_path):
            target_name = dst_name if dst_name else os.path.basename(src_path)
            target_path = os.path.join(dst_dir, target_name)
            # Remove target if exists to avoid shutil.move conflicts
            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.move(src_path, target_path)
            #print(f"# [BATCH_DEBUG] moved '{src_path}' -> '{target_path}'")
        else:
            print(f"# Warning: expected file not found, skipping move: {src_path}")

    def _copy_if_exists(src_path, dst_dir, dst_name=None):
        """Copy a file from src_path to dst_dir, optionally renaming to dst_name."""
        if src_path is None:
            #print(f"# [BATCH_DEBUG] copy skipped (source is None) -> dst_dir={dst_dir}")
            return
        if os.path.isfile(src_path):
            target_name = dst_name if dst_name else os.path.basename(src_path)
            target_path = os.path.join(dst_dir, target_name)
            shutil.copy2(src_path, target_path)
            #print(f"# [BATCH_DEBUG] copied '{src_path}' -> '{target_path}'")
        else:
            print(f"# Warning: expected file not found, skipping copy: {src_path}")

    # Scan existing SIMXXXXX folders and allocate new sequential indices
    sim_root = os.getcwd()
    #print(f"# [BATCH_DEBUG] scanning existing SIM dirs in: {sim_root}")
    existing = []
    for name in os.listdir(sim_root):
        m = re.fullmatch(r"SIM(\d{5})", name)
        if m and os.path.isdir(os.path.join(sim_root, name)):
            existing.append(int(m.group(1)))
    # Allocate next batch: if folders exist, start after the highest; else start at 0
    start_sim = (max(existing) + 1) if existing else 0
    if existing:
        #print(f"# [BATCH_DEBUG] existing SIM indices: {sorted(existing)}")
        pass
    else:
        #print("# [BATCH_DEBUG] no existing SIM dirs found; starting at SIM00000")
        pass
    #print(f"# [BATCH_DEBUG] allocated start index: {start_sim:05d}")

    # Create SIM folders for this batch (batch_size of them)
    sim_dirs = []
    for i in range(batch_size):
        dname = f"SIM{start_sim + i:05d}"
        dpath = os.path.join(sim_root, dname)
        os.makedirs(dpath, exist_ok=True)
        sim_dirs.append(dpath)
    #print(f"# [BATCH_DEBUG] created/verified SIM dirs: {[os.path.basename(d) for d in sim_dirs]}")

    # Resolve paths to input files (initial geometry, input YAML) for copying
    input_file_abs = os.path.abspath(args.input_file)
    #print(f"# [BATCH_DEBUG] input file resolved to: {input_file_abs}")
    initial_geom_abs = None
    if isinstance(initial_xyz, str):
        # Try to resolve initial geometry file in input dir or current dir
        initial_geom_abs = _resolve_existing_path(
            initial_xyz,
            [input_yaml_dir, os.getcwd()]
        )
    #print(f"# [BATCH_DEBUG] initial geometry resolved to: {initial_geom_abs}")

    # Prepare per-trajectory summary file sources (per-trajectory summary files from main run)
    per_summary_sources = None
    if isinstance(summary_files, list):
        # summary_files contains the actual per-trajectory summary output files
        per_summary_sources = summary_files
    #print(f"# [BATCH_DEBUG] per_summary_sources: {per_summary_sources}")

    # Destination base names: files copied into SIM folders are renamed to configured base names (no _i suffix)
    # This makes each SIM folder independent and self-contained
    traj_dst_name = os.path.basename(traj_file) if write_traj else None
    if isinstance(energy_file, str):
        energy_dst_name = os.path.basename(energy_file)
    else:
        energy_dst_name = None
    if isinstance(summary_cfg, str):
        summary_dst_name = os.path.basename(summary_cfg)
    else:
        summary_dst_name = None
    print(
    #    f"# [BATCH_DEBUG] destination base names -> "
        f"traj: {traj_dst_name}, energy: {energy_dst_name}, summary: {summary_dst_name}"
    )

    # Move/copy per-trajectory outputs into their respective SIM folders
    for i, sim_dir in enumerate(sim_dirs):
        #print(f"# [BATCH_DEBUG] processing trajectory index {i} for '{sim_dir}'")
        
        # Move trajectory file (generated output with _i suffix) -> destination with base name
        if write_traj and i < len(traj_paths):
            _move_if_exists(os.path.abspath(traj_paths[i]), sim_dir, traj_dst_name)

        # Move energy file (generated output with _i suffix) -> destination with base name
        if energy_file and i < len(energy_files):
            if energy_dst_name:
                _move_if_exists(os.path.abspath(energy_files[i]), sim_dir, energy_dst_name)
            else:
                # If energy_file is a custom list, keep source basename
                _move_if_exists(os.path.abspath(energy_files[i]), sim_dir)

        # Move summary file (per-trajectory summary from main run) -> destination with base name
        if per_summary_sources is not None and i < len(per_summary_sources):
            if summary_dst_name:
                _move_if_exists(os.path.abspath(per_summary_sources[i]), sim_dir, summary_dst_name)
            else:
                _move_if_exists(os.path.abspath(per_summary_sources[i]), sim_dir)

        # Always copy input YAML and starting geometry 
        _copy_if_exists(input_file_abs, sim_dir)
        _copy_if_exists(initial_geom_abs, sim_dir)

    # Report completion of batch artifact export
    if batch_size > 1:
        print(
            f"# Exported batch artifacts into {batch_size} simulation folder(s): "
            f"SIM{start_sim:05d}..SIM{start_sim + batch_size - 1:05d}"
        )
        #print("# [BATCH_DEBUG] batch artifact export phase completed")


if __name__ == "__main__":
    main()




