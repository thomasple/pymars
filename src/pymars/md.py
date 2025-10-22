import numpy as np
from fennol.utils.periodic_table import PERIODIC_TABLE, ATOMIC_MASSES
from .utils import (
    us,
    get_composition_string,
    format_batch_conformations,
    update_conformation,
)
from .rotation_utils import apply_random_rotation
from .repulsion import setup_repulsion_potential
from .initial_configuration import (
    read_initial_configuration,
    sample_velocities,
    remove_com_velocity,
    sample_projectiles,
)
from pathlib import Path


def initialize_collision_simulation(simulation_parameters, verbose=True):

    # load initial configuration
    geometry = simulation_parameters["initial_geometry"]
    if isinstance(geometry, str):
        assert Path(geometry).is_file(), f"File {geometry} does not exist."
        # load from file
        species, coordinates = read_initial_configuration(geometry)
    elif isinstance(geometry, dict):
        # load directly from dict
        species = np.array(geometry["species"], dtype=np.int32)
        coordinates = np.array(geometry["coordinates"], dtype=np.float32).reshape(-1, 3)
    else:
        raise ValueError("initial_geometry must be a file path or a dictionary")

    total_charge = round(simulation_parameters.get("total_charge", 0))
    if verbose:
        print(f"# total charge of the system: {total_charge} e")

    if verbose:
        composition_str = get_composition_string(species)
        print("# Initial configuration loaded: ", composition_str)

    # batch size for parallel simulations
    batch_size = simulation_parameters.get("batch_size", 1)
    assert batch_size >= 1, "batch_size must be at least 1"

    # random rotation
    do_random_rotation = simulation_parameters.get("random_rotation", True)
    if do_random_rotation:
        coordinates = apply_random_rotation(coordinates, n_rotations=batch_size)
        if verbose:
            print("# Applied random rotation to initial configuration.")
    else:
        coordinates = np.tile(coordinates, (batch_size, 1, 1))  # (batch_size,N,3)

    batch_species = np.tile(species, batch_size)  # (batch_size*N)

    # sample velocities from Maxwell-Boltzmann distribution
    temperature = simulation_parameters.get("temperature", 0.0)  # Kelvin
    assert temperature >= 0.0, "Temperature must be non-negative"
    velocities = sample_velocities(batch_species, temperature).reshape(
        batch_size, -1, 3
    )  # (batch_size,N,3)
    if verbose:
        print(f"# Sampled velocities at T={temperature} K.")

    # remove center of mass linear and angular velocities
    for b in range(batch_size):
        velocities[b] = remove_com_velocity(coordinates[b], velocities[b], species)
    if verbose:
        print("# Removed center of mass linear and angular velocities.")

    # coordinates bounding box
    molecule_radius = np.max(np.linalg.norm(coordinates, axis=-1))  # angstrom

    # initialize projectiles
    projectile_species = simulation_parameters.get(
        "projectile_species", 18
    )  # default Argon
    max_impact_parameter = simulation_parameters.get(
        "max_impact_parameter", 0.5
    )  # angstrom
    projectile_distance = simulation_parameters.get(
        "projectile_distance", 10.0 + 2 * molecule_radius
    )  # angstrom
    assert (
        projectile_distance > 2 * molecule_radius
    ), f"projectile_distance must be larger than twice molecule radius ({2*molecule_radius:.2f} A)"

    projectile_temperature = simulation_parameters.get(
        "projectile_temperature", temperature
    )  # Kelvin
    assert projectile_temperature > 0.0, "projectile_temperature must > 0 K"
    projectile_coordinates, projectiles_velocities = sample_projectiles(
        batch_size,
        temperature=projectile_temperature,
        distance=projectile_distance,
        projectile_species=projectile_species,
        max_impact_parameter=max_impact_parameter,
    )

    if verbose:
        projectile_vel = np.linalg.norm(projectiles_velocities[0])
        distance_to_impact = projectile_distance - molecule_radius
        time_to_impact = us.PS * distance_to_impact / projectile_vel
        print(
            f"# initialized projectile at distance {projectile_distance:.2f} A with temperature {projectile_temperature} K"
        )
        print(f"# Time before collision: ~{time_to_impact:.2f} ps")

    model_file = simulation_parameters["model"]
    assert isinstance(model_file, str), "model must be a string"
    if model_file.upper()[:3] in ["GFN", "XTB"]:
        import torch
        import dxtb

        opts = {"batch_mode": 2}
        torch_device = simulation_parameters.get("torch_device", "cpu")
        dd = {"dtype": torch.float32, "device": torch.device(torch_device)}
        species_torch = torch.tensor(
            batch_species.reshape(batch_size, -1), device=dd["device"]
        )
        if model_file.upper() in ["GFN2", "GFN2-XTB", "GFN2XTB", "XTB"]:
            print("# Using GFN2-XTB model.")
            model = dxtb.calculators.GFN2Calculator(species_torch, opts=opts,**dd)
            from .gfn_references import GFN2_REF_ENERGIES
            ref_energy = GFN2_REF_ENERGIES[species].sum()
        elif model_file.upper() in ["GFN1", "GFN1-XTB", "GFN1XTB"]:
            print("# Using GFN1-XTB model.")
            model = dxtb.calculators.GFN1Calculator(species_torch, opts=opts,**dd)
            from .gfn_references import GFN1_REF_ENERGIES
            ref_energy = GFN1_REF_ENERGIES[species].sum()
        else:
            raise ValueError(f"Unknown GFN-XTB model: {model_file}")

        dxtb.OutputHandler.verbosity = 0  # suppress output
        energy_conv = 1.0 / us.HARTREE
        force_conv = -1.0 / (us.HARTREE / us.BOHR)
        charge = torch.tensor(
            [total_charge] * batch_size, device=dd["device"], dtype=torch.float32
        ).reshape(batch_size, 1)

        def model_energies_and_forces(coordinates):
            model.reset()
            coords_torch = torch.tensor(coordinates * us.BOHR, **dd).requires_grad_(
                True
            )
            energies = model.get_energy(coords_torch, chrg=charge)
            dedx = torch.autograd.grad(energies.sum(), coords_torch)[0]
            return (
                (energies.detach().cpu().numpy()-ref_energy) * energy_conv,
                dedx.detach().cpu().numpy() * force_conv,
            )

    else:
        assert Path(model_file).is_file(), f"Model file {model_file} does not exist."
        print(f"# Using FENNIX model from file: {model_file}")
        from fennol import FENNIX

        model = FENNIX.load(model_file)
        model.preproc_state = model.preproc_state.copy({"check_input": False})
        energy_conv = 1.0 / us.get_multiplier(model.energy_unit)

        initial_conformation = format_batch_conformations(
            species, coordinates, total_charge=total_charge
        )

        def model_energies_and_forces(coordinates):
            conformation = update_conformation(initial_conformation, coordinates)
            energies, forces, _ = model.energy_and_forces(
                **conformation, gpu_preprocessing=True
            )
            return (
                np.array(energies) * energy_conv,
                np.array(forces).reshape(batch_size, -1, 3) * energy_conv,
            )

    repulsion_energies_and_forces = setup_repulsion_potential(
        species, projectile_species
    )

    def total_energies_and_forces(full_coordinates):
        coordinates = full_coordinates[:, 1:, :]
        projectile_coordinates = full_coordinates[:, 0, :]
        energies_model, forces_model = model_energies_and_forces(coordinates)
        energies_repulsion, forces_repulsion, projectile_forces = (
            repulsion_energies_and_forces(coordinates, projectile_coordinates)
        )
        total_energies = energies_model + energies_repulsion
        total_forces = forces_model + forces_repulsion

        full_forces = np.concatenate(
            [projectile_forces[:, None, :], total_forces], axis=1
        )  # (batch_size,N+1,3)
        return total_energies, full_forces

    full_species = np.concatenate(
        [np.array([projectile_species], dtype=np.int32), species]
    )  # (N+1,)
    full_coordinates = np.concatenate(
        [projectile_coordinates[:, None, :], coordinates], axis=1
    )  # (batch_size,N+1,3)
    full_velocities = np.concatenate(
        [projectiles_velocities[:, None, :], velocities], axis=1
    )  # (batch_size,N+1,3)

    # compute initial accelerations
    masses = (
        ATOMIC_MASSES[full_species].astype(np.float32)[None, :, None] / us.DA
    )  # (N,)

    energies, forces = total_energies_and_forces(full_coordinates)
    accelerations = forces / masses  # (batch_size,N,3)

    # prepare integrator
    dt = simulation_parameters.get("dt", 1.0 / us.FS)
    dt2 = dt * 0.5

    def integrate(initial_coordinates, initial_velocities, accelerations):
        # Velocity Verlet step
        velocities = initial_velocities + accelerations * dt2  # (batch_size,N+1,3)
        coordinates = initial_coordinates + velocities * dt  # (batch_size,N+1,3)

        energies, forces = total_energies_and_forces(coordinates)
        accelerations = forces / masses  # (batch_size,N+1,3)

        velocities += accelerations * dt2  # (batch_size,N,3)

        return coordinates, velocities, accelerations, energies

    return {
        "species": full_species,
        "masses": masses.reshape(-1),
        "coordinates": full_coordinates,
        "velocities": full_velocities,
        "accelerations": accelerations,
        "total_energies_and_forces": total_energies_and_forces,
        "integrate": integrate,
        "batch_size": batch_size,
        "dt": dt,
    }
