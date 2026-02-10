import argparse
import yaml
import numpy as np
import os
import jax
import jax.numpy as jnp
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

    from fennol.utils.input_parser import convert_dict_units
    from .utils import us
    simulation_parameters = convert_dict_units(simulation_parameters, us)

    ### Set the device
    if "MARS_DEVICE" in os.environ:
        device = os.environ["MARS_DEVICE"].lower()
        print(f"# Setting device from env MARS_DEVICE={device}")
    else:
        device = simulation_parameters.get("device", "cpu").lower()
        """@keyword[fennol_md] device
        Computation device. Options: 'cpu', 'cuda:N', 'gpu:N' where N is device number.
        Default: 'cpu'
        """
    if device == "cpu":
        jax.config.update("jax_platforms", "cpu")
        jax.config.update("jax_cuda_visible_devices", "")
        simulation_parameters["torch_device"] = "cpu"
    elif device.startswith("cuda") or device.startswith("gpu"):
        if ":" in device:
            num = device.split(":")[-1]
            jax.config.update("jax_cuda_visible_devices", num)
        else:
            jax.config.update("jax_cuda_visible_devices", "0")
        device = "gpu"
        simulation_parameters["torch_device"] = "cuda:0"

    _device = jax.devices(device)[0]
    jax.config.update("jax_default_device", _device)
    jax.config.update("jax_default_matmul_precision", "highest")

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
    if "step_dyn" in dynamic_params:
        n_steps = int(dynamic_params["step_dyn"])
        simulation_time = n_steps * dt
    elif "simulation_time" in simulation_parameters:
        simulation_time = simulation_parameters["simulation_time"]  # ps
        n_steps = round(simulation_time / dt)
        simulation_time = n_steps * dt  # adjust to exact number of steps
    else:
        raise ValueError("Either step_dyn or simulation_time must be specified")
    
    print(f"# Running simulation for {simulation_time} ps ({n_steps} steps of {dt} ps)")

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
    print(f"#{'Step':>10} {'Time':>12} {'Etot':>12} {'Epot':>12} {'Ekin':>12} {'ns/day':>12}")
    for istep in range(n_steps):
        coordinates, velocities, accelerations, energies = integrate(
            coordinates, velocities, accelerations,
            step=istep,
            energy_output_file=energy_files if energy_file else None,
            energy_steps=save_energy
        )
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

            print(
                f" {istep+1:10} {(istep+1)*dt:12.2f} {total_energy:12.3f} {potential_energy:12.3f} {ekin:12.3f} {ns_per_day:12.1f}"
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

    if write_traj:
        for f in ftraj: 
            f.close()
    from fennol.utils.io import human_time_duration
    total_time = time.time() - time_start
    nsperday = (simulation_time / total_time)*60*60*24*us.NS
    print(f"# {simulation_time*us.PS} ps simulation completed in {human_time_duration(total_time)} ({nsperday:.1f} ns/day)")


if __name__ == "__main__":
    main()




