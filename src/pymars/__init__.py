import argparse
import yaml
import numpy as np


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

    from .md import initialize_collision_simulation
    system = initialize_collision_simulation(simulation_parameters)

    integrate = system["integrate"]
    coordinates = system["coordinates"]
    velocities = system["velocities"]
    accelerations = system["accelerations"]
    species = system["species"]
    masses = system["masses"]
    batch_size = system["batch_size"]

    simulation_time = simulation_parameters["simulation_time"]  # ps
    dt = system["dt"]  # ps
    n_steps = round(simulation_time / dt)
    simulation_time = n_steps * dt  # adjust to exact number of steps
    print(f"# Running simulation for {simulation_time} ps ({n_steps} steps of {dt} ps)")

    print_step = simulation_parameters.get("print_step", 100)

    output_file = simulation_parameters.get("output_file", "trajectory.xyz")
    from fennol.utils.io import write_xyz_frame
    ftraj = open(output_file, "w")
    import time
    time_start = time.time()
    time0 = time_start
    print(f"#{'Step':>10} {'Time':>12} {'Etot':>12} {'Epot':>12} {'Ekin':>12} {'ns/day':>12}")
    for istep in range(n_steps):
        coordinates, velocities, accelerations, energies = integrate(
            coordinates, velocities, accelerations
        )
        if (istep + 1) % print_step == 0:
            time_elapsed = time.time() - time0
            time0 = time.time()
            time_per_step = time_elapsed / print_step
            step_per_day = 60*60*24 / time_per_step
            ns_per_step = dt*us.NS
            ns_per_day = ns_per_step * step_per_day

            ekin = 0.5 * np.sum(
                masses[None,:] * np.sum(velocities**2, axis=-1)
            ) / batch_size  # kcal/mol
            potential_energy = np.mean(energies)
            total_energy = ekin + potential_energy
            print(
                f" {istep+1:10} {(istep+1)*dt:12.2f} {total_energy:12.3f} {potential_energy:12.3f} {ekin:12.3f} {ns_per_day:12.1f}"
            )
            com = np.mean(coordinates[:,1:,:], axis=1, keepdims=True)
            coordinates -= com  # center at COM
            write_xyz_frame(
                ftraj,
                species,
                coordinates[0],  # write first trajectory only
                comment=f"Step {istep+1} E_tot={total_energy:.6f} E_pot={potential_energy:.6f} E_kin={ekin:.6f}",
            )

    ftraj.close()
    from fennol.utils.io import human_time_duration
    total_time = time.time() - time_start
    nsperdat = (simulation_time / total_time)*60*60*24*us.NS
    print(f"# {simulation_time*us.PS} ps simulation completed in {human_time_duration(total_time)} ({nsperdat:.1f} ns/day)")


if __name__ == "__main__":
    main()




