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
import jax.numpy as jnp
import jax


def initialize_collision_simulation(simulation_parameters, verbose=True):
    """
    Initialize collision (or molecule-only) simulation.

    Returns a dict with:
      - species, masses (numpy), coordinates (jnp), velocities (jnp), accelerations (jnp)
      - total_energies_and_forces callable
      - integrate(...) function to run dynamics
      - batch_size, dt
    """

    # Extract nested parameter sections
    input_params = simulation_parameters.get("input_parameters", {})
    general_params = simulation_parameters.get("general_parameters", {})
    projectile_params = simulation_parameters.get("projectile_parameters", {})
    calculation_params = simulation_parameters.get("calculation_parameters", {})
    dynamic_params = simulation_parameters.get("dynamic_parameters", {})
    
    # Support both nested and flat (legacy) formats
    # If nested sections don't exist, fall back to top-level keys
    if not input_params and not general_params:
        # Legacy flat format - use simulation_parameters directly
        input_params = simulation_parameters
        general_params = simulation_parameters
        projectile_params = simulation_parameters
        calculation_params = simulation_parameters
        dynamic_params = simulation_parameters
    
    # Determine if this is collision dynamics or molecule-only
    # Check projectile_flag in nested format, or collision_dynamics in flat format
    if "projectile_flag" in projectile_params:
        collision = bool(projectile_params.get("projectile_flag", True))
    else:
        # Fall back to collision_dynamics (legacy)
        coll_dyn = simulation_parameters.get("collision_dynamics", True)
        if isinstance(coll_dyn, str):
            collision = coll_dyn.strip().upper() in ("TRUE", "YES", "1")
        else:
            collision = bool(coll_dyn)

    # load initial configuration
    geometry = input_params.get("initial_geometry", simulation_parameters.get("initial_geometry"))
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

    total_charge = round(input_params.get("total_charge", 0))
    if verbose:
        print(f"# total charge of the system: {total_charge} e")

    if verbose:
        composition_str = get_composition_string(species)
        print("# Initial configuration loaded: ", composition_str)

    # batch size for parallel simulations
    batch_size = general_params.get("batch_size", 1)
    assert batch_size >= 1, "batch_size must be at least 1"

    # random rotation
    do_random_rotation = input_params.get("random_rotation", True)
    if do_random_rotation:
        coordinates = apply_random_rotation(coordinates, n_rotations=batch_size)
        if verbose:
            print("# Applied random rotation to initial configuration.")
    else:
        coordinates = np.tile(coordinates, (batch_size, 1, 1))  # (batch_size,N,3)

    batch_species = np.tile(species, batch_size)  # (batch_size*N)

    # sample velocities from Maxwell-Boltzmann distribution
    temperature = general_params.get("temperature", 0.0)  # Kelvin
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

    if collision:
        # initialize projectiles
        projectile_species = int(projectile_params.get("projectile_species", 18))  # default Argon
        max_impact_parameter = float(projectile_params.get("max_impact_parameter", 0.5))  # angstrom
        projectile_distance = float(projectile_params.get(
            "projectile_distance", 10.0 + 2 * molecule_radius
        ))  # angstrom
        assert (
            projectile_distance > 2 * molecule_radius
        ), f"projectile_distance must be larger than twice molecule radius ({2*molecule_radius:.2f} A)"

        projectile_temperature = float(projectile_params.get(
            "projectile_temperature", temperature
        ))  # Kelvin
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

    # Support both "model" and "model_file" keys for backward compatibility
    # Try nested config first (calculation_parameters), then fall back to flat config
    model_file = calculation_params.get("model") or calculation_params.get("model_file")
    if model_file is None:
        model_file = simulation_parameters.get("model") or simulation_parameters.get("model_file")
    if model_file is None:
        raise ValueError("Configuration must contain either 'model_file' or 'model' key")
    
    model_path = Path(model_file)
    assert model_path.is_file(), f"Model file {model_file} does not exist."
    print(f"# Using FENNIX model from file: {model_file}")
    
    # Load FeNNol model (FENNOL_MODULES_PATH already set in __init__.py)
    from fennol import FENNIX
    model = FENNIX.load(model_file)
    model.preproc_state = model.preproc_state.copy({"check_input": False})
    energy_conv = 1.0 / us.get_multiplier(model.energy_unit)

    initial_conformation = format_batch_conformations(
        species, coordinates, total_charge=total_charge
    )

    if collision:
        repulsion_energies_and_forces = setup_repulsion_potential(
            species, projectile_species, use_jax=True
        )

        def total_energies_and_forces(full_coordinates, conformation):
            # full_coordinates expected shape (batch_size, N+1, 3) as jnp array
            coordinates_model = full_coordinates[:, 1:, :]
            projectile_coordinates = full_coordinates[:, 0, :]
            energies_model, forces_model, _ = model._energy_and_forces(model.variables, conformation)
            energies_repulsion, forces_repulsion, projectile_forces = (
                repulsion_energies_and_forces(coordinates_model, projectile_coordinates)
            )
            total_energies = energies_model * energy_conv + energies_repulsion
            total_forces = forces_model.reshape(coordinates_model.shape[0], -1, 3) * energy_conv + forces_repulsion

            full_forces = jnp.concatenate(
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

    else:
        def total_energies_and_forces(full_coordinates, conformation):
            # full_coordinates shape (batch_size, N, 3)
            coordinates_model = full_coordinates[:, :, :]
            energies_model, forces_model, _ = model._energy_and_forces(model.variables, conformation)
            total_energies = energies_model * energy_conv
            total_forces = forces_model.reshape(coordinates_model.shape[0], -1, 3) * energy_conv
            full_forces = total_forces  # (batch_size,N,3)
            return total_energies, full_forces

        full_species = species.copy()  # (N,)
        full_coordinates = coordinates.copy()  # (batch_size,N,3)
        full_velocities = velocities.copy()  # (batch_size,N,3)

    # compute initial accelerations
    masses_np = ATOMIC_MASSES[full_species].astype(np.float32) / us.DA  # (N,) in atomic units
    masses = jnp.array(masses_np)[None, :, None]  # (1,N,1) for broadcasting in jax

    # Preprocess initial conformation for the model
    conformation = model.preprocess(use_gpu=True, **initial_conformation)
    # Ensure full_coordinates passed as jnp arrays to energy/force function
    full_coordinates_jnp = jnp.array(full_coordinates, dtype=jnp.float32)
    energies, forces = total_energies_and_forces(full_coordinates_jnp, conformation)
    accelerations = forces / masses  # (batch_size, N(+1), 3) depending on collision

    # prepare integrator
    # dt_dyn is given in fs in the input, convert to ps (internal unit)
    dt_fs = dynamic_params.get("dt_dyn", dynamic_params.get("dt", 1.0))  # fs
    dt = dt_fs / 1000.0  # Convert fs to ps
    dt2 = dt * 0.5

    @jax.jit
    def integrate_part1(coordinates, velocities, accelerations):
        # Velocity Verlet step - part 1
        velocities = velocities + accelerations * dt2  # (batch_size,N+1,3)
        coordinates = coordinates + velocities * dt  # (batch_size,N+1,3)
        return coordinates, velocities
    
    @jax.jit
    def integrate_part2(coordinates, velocities,conformation):
        energies, forces = total_energies_and_forces(coordinates,conformation)
        accelerations = forces / masses  # (batch_size,N+1,3)

        velocities = velocities + accelerations * dt2  # (batch_size,N,3)
        return velocities, accelerations, energies  # energies is (batch_size,)

    def integrate(initial_coordinates, initial_velocities, accelerations, step=0, energy_output_file=None, energy_steps=100):
        # Velocity Verlet step
        coordinates, velocities = integrate_part1(
            initial_coordinates, initial_velocities, accelerations
        )
        
        # Update conformation for model preprocessing
        if collision:
            # For collision dynamics, exclude projectile (index 0)
            coords_for_model = coordinates[:,1:,:]
        else:
            # For non-collision dynamics, use all coordinates
            coords_for_model = coordinates
            
        conformation = model.preprocess(use_gpu=True,**update_conformation(
            initial_conformation, coords_for_model))

        velocities, accelerations, energies = integrate_part2(
            coordinates, velocities,conformation
        )

        # Write energy output if requested
        energy_data = None
        if energy_output_file is not None and step % energy_steps == 0:
            energy_data = write_energy_output(
                energy_output_file, step, velocities, masses, energies, dt
            )

        return coordinates, velocities, accelerations, energies, energy_data

    def write_energy_output(output_file, step, velocities, masses, potential_energies, dt):
        """Write energy data to output file in FeNNol format."""
        # Compute kinetic energy: 0.5 * m * v^2
        # masses is already shaped (1, N, 1) for broadcasting
        kinetic_energies = 0.5 * jnp.sum(masses * velocities**2, axis=(1, 2))
        
        # Ensure potential energies are 1D (batch_size,)
        potential_energies = jnp.squeeze(potential_energies)
        kinetic_energies = jnp.squeeze(kinetic_energies)
        
        # Total energy
        total_energies = potential_energies + kinetic_energies
        
        # Temperature: T = (2 * Ekin) / (3 * N * k_B)
        # k_B in kcal/(mol*K)
        k_B = 0.0019872043  # kcal/(mol*K)
        N_atoms = velocities.shape[1]
        temperatures = (2.0 * kinetic_energies) / (3.0 * N_atoms * k_B)
        
        # Time in femtoseconds - determine precision based on dt
        time_fs = step * dt * 1000.0  # dt is in ps, convert to fs
        # Determine decimal places: if dt=1.0 fs, show 0 decimals; if dt=0.1, show 1, etc.
        dt_fs = dt * 1000.0  # Convert dt from ps to fs
        if dt_fs >= 1.0:
            time_decimals = 0
        else:
            # Count decimal places needed
            time_decimals = len(str(dt_fs).split('.')[-1].rstrip('0'))
        time_format = f"{{:.{time_decimals}f}}"
        
        # Convert JAX arrays to numpy for file writing
        total_energies_np = np.array(total_energies)
        potential_energies_np = np.array(potential_energies)
        kinetic_energies_np = np.array(kinetic_energies)
        temperatures_np = np.array(temperatures)
        
        # Ensure arrays are at least 1D for concatenation in summary statistics
        total_energies_np = np.atleast_1d(total_energies_np)
        potential_energies_np = np.atleast_1d(potential_energies_np)
        kinetic_energies_np = np.atleast_1d(kinetic_energies_np)
        temperatures_np = np.atleast_1d(temperatures_np)
        
        # Write to file for each trajectory in batch
        for b in range(batch_size):
            if isinstance(output_file, list):
                file_path = output_file[b]
            else:
                file_path = output_file if batch_size == 1 else f"{output_file}_{b}"
            
            # Create file with header if step == 0
            if step == 0:
                with open(file_path, 'w') as f:
                    f.write(f"{'Step':>8s} {'Time[fs]':>12s} {'Etot':>12s} {'Epot':>12s} {'Ekin':>12s} {'Temp[K]':>10s}\n")
            
            # Append energy data
            with open(file_path, 'a') as f:
                # Extract scalar values properly
                total_e = float(total_energies_np.flat[b] if total_energies_np.size > 1 else total_energies_np)
                pot_e = float(potential_energies_np.flat[b] if potential_energies_np.size > 1 else potential_energies_np)
                kin_e = float(kinetic_energies_np.flat[b] if kinetic_energies_np.size > 1 else kinetic_energies_np)
                temp = float(temperatures_np.flat[b] if temperatures_np.size > 1 else temperatures_np)
                
                # Format time with appropriate precision
                time_str = time_format.format(time_fs)
                f.write(f"{step:8d} {time_str:>12s} {total_e:12.6f} "
                       f"{pot_e:12.6f} {kin_e:12.6f} "
                       f"{temp:10.2f}\n")
        
        # Return energy data for summary statistics
        return {
            'total_energies': total_energies_np,
            'potential_energies': potential_energies_np,
            'kinetic_energies': kinetic_energies_np,
            'temperatures': temperatures_np
        }

    return {
        "species": full_species,
        "masses": masses_np,
        "coordinates": full_coordinates_jnp,
        "velocities": jnp.array(full_velocities,dtype=jnp.float32),
        "accelerations": accelerations,
        "total_energies_and_forces": total_energies_and_forces,
        "integrate": integrate,
        "batch_size": batch_size,
        "dt": dt,
        "initial_energies": energies,
    }
