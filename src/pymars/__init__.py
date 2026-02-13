
import argparse
import yaml
import numpy as np
import os
import time


__all__ = []

def main() -> None:
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
    # Now import jax and set configs
    import jax
    import jax.numpy as jnp

    if device == "cpu":
        jax.config.update("jax_platforms", "cpu")
        jax.config.update("jax_cuda_visible_devices", "")
        jax.config.update("jax_platforms", "cpu")
        jax.config.update("jax_cuda_visible_devices", "")
        simulation_parameters["torch_device"] = "cpu"
        jax_device_str = "cpu"
    elif device.startswith("cuda") or device.startswith("gpu"):
        jax.config.update("jax_platforms", "")
        if ":" in device:
            num = device.split(":")[-1]
            jax.config.update("jax_cuda_visible_devices", num)
            jax.config.update("jax_cuda_visible_devices", num)
        else:
            jax.config.update("jax_cuda_visible_devices", "0")
        device = "gpu"
        simulation_parameters["torch_device"] = "cuda:0"
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
    
    ### Set random seed for reproducibility
    general_params = simulation_parameters.get("general_parameters", simulation_parameters)
    random_seed = general_params.get("seed", np.random.randint(0, 2**32 - 1))
    print(f"# Random seed: {random_seed}")
    np.random.seed(random_seed)
    # JAX random key will be set in md.py for velocity sampling
    
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
    coordinates = system["coordinates"]
    velocities = system["velocities"]
    accelerations = system["accelerations"]
    species = system["species"]
    masses = system["masses"]
    batch_size = system["batch_size"]
    
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
    write_traj = traj_file.lower() != "none"
    if write_traj:
        assert traj_file.endswith(".xyz"), "Only .xyz output format is supported currently."
        from fennol.utils.io import write_xyz_frame
        if batch_size ==1:
            ftraj = [open(traj_file, "w")]
        else:
            num=len(str(batch_size-1))
            ftraj = [open(traj_file.replace(".xyz",f'.{i:0{num}}.xyz'), "w") for i in range(batch_size)]
    
    # Get energy output file
    energy_file = output_params.get("energies_file", output_params.get("energy_output_file", None))
    if energy_file:
        if isinstance(energy_file, str):
            if batch_size == 1:
                energy_files = [energy_file]
            else:
                energy_files = [energy_file.replace(".out", f"_{i}.out") for i in range(batch_size)]
        else:
            energy_files = energy_file

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
    
    # Prepare summary file if requested
    summary_file = output_params.get("summary_file", "summary.out") if save_summary else None
    
    # Energy tracking for summary and drift calculation
    initial_total_energy = None
    max_energy_drift = 0.0
    energy_history = []  # Store energy data for summary statistics
    
    print(f"#{'Step':>10} {'Time[fs]':>12} {'Etot':>12} {'Epot':>12} {'Ekin':>12} {'ns/day':>12}")
    for istep in range(n_steps):
        coordinates, velocities, accelerations, energies, energy_data = integrate(
            coordinates, velocities, accelerations,
            step=istep,
            energy_output_file=energy_files if energy_file else None,
            energy_steps=save_energy
        )
        
        # Collect energy data for summary statistics
        if energy_data is not None:
            energy_history.append(energy_data)
            # Set initial energy for drift calculation
            if initial_total_energy is None:
                initial_total_energy = np.mean(energy_data['total_energies'])
            # Track maximum energy drift
            current_total_energy = np.mean(energy_data['total_energies'])
            drift = abs(current_total_energy - initial_total_energy)
            max_energy_drift = max(max_energy_drift, drift)
        
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
            
            print(
                f" {istep+1:10} {time_str:>12} {total_energy:12.3f} {potential_energy:12.3f} {ekin:12.3f} {ns_per_day:12.1f}"
            )
            if write_traj:
                coords = np.array(coordinates)
                for i in range(batch_size):
                    write_xyz_frame(
                        ftraj[i],
                        element_symbols,
                        coords[i],
                        comment=f"Step {istep+1} E_pot={potential_energy:.6f}",
                    )
        
        # Write summary output if requested
        if save_summary is not None and summary_file is not None and (istep + 1) % save_summary == 0:
            from fennol.utils.io import human_time_duration
            
            # Calculate statistics over the summary interval
            n_summary_steps = len(energy_history)
            if n_summary_steps > 0:
                # Aggregate energy data
                all_etot = np.concatenate([e['total_energies'] for e in energy_history])
                all_epot = np.concatenate([e['potential_energies'] for e in energy_history])
                all_ekin = np.concatenate([e['kinetic_energies'] for e in energy_history])
                all_temp = np.concatenate([e['temperatures'] for e in energy_history])
                
                # Calculate averages and standard deviations
                avg_etot = np.mean(all_etot)
                std_etot = np.std(all_etot)
                avg_epot = np.mean(all_epot)
                std_epot = np.std(all_epot)
                avg_ekin = np.mean(all_ekin)
                std_ekin = np.std(all_ekin)
                avg_temp = np.mean(all_temp)
                std_temp = np.std(all_temp)
                
                # Calculate energy drift percentage
                drift_percent = (max_energy_drift / abs(initial_total_energy)) * 100.0 if initial_total_energy != 0 else 0.0
                
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
                
                # Write summary to file
                with open(summary_file, 'a') as f:
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
                        f.write(f"# Maximum energy drift (NVE): {max_energy_drift:.2f} kcal/mol ({drift_percent:.1f}%)\n")
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
    from fennol.utils.io import human_time_duration
    total_time = time.time() - time_start
    nsperday = (simulation_time / total_time)*60*60*24*us.NS
    print(f"# {simulation_time*us.PS} ps simulation completed in {human_time_duration(total_time)} ({nsperday:.1f} ns/day)")


if __name__ == "__main__":
    main()




