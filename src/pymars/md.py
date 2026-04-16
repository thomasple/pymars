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
    Initialize collision (or molecule-only) simulation with batch processing support.
    
    Handles parameter parsing (nested + legacy formats), loads model and initial geometry,
    sets up per-trajectory or per-projectile dynamics with optional repulsion potential,
    and returns integrator function + initial state for main loop.

    Returns a dict with:
      - species, masses (numpy), coordinates (jnp), velocities (jnp), accelerations (jnp)
      - total_energies_and_forces callable
      - integrate(...) function to run dynamics
      - batch_size, dt
    """

    # Extract nested parameter sections (new config format)
    # Falls back to flat structure if nested keys missing (legacy format compatibility)
    input_params = simulation_parameters.get("input_parameters", {})
    general_params = simulation_parameters.get("general_parameters", {})
    projectile_params = simulation_parameters.get("projectile_parameters", {})
    calculation_params = simulation_parameters.get("calculation_parameters", {})
    dynamic_params = simulation_parameters.get("dynamic_parameters", {})
    # Output options
    output_params = simulation_parameters.get("output_details", {})
    track_variance = bool(output_params.get("track_variance", False))
    # Variance tracking requires batch_size==1 (model etot_ensemble_var is per-frame, not across trajectories)
    # The guard below will be enforced after batch_size is known.
    
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
    # Collision mode adds a projectile particle; non-collision targets molecule only
    if "projectile_flag" in projectile_params:
        collision = bool(projectile_params.get("projectile_flag", True))
    else:
        # Fall back to collision_dynamics (legacy)
        coll_dyn = simulation_parameters.get("collision_dynamics", True)
        if isinstance(coll_dyn, str):
            collision = coll_dyn.strip().upper() in ("TRUE", "YES", "1")
        else:
            collision = bool(coll_dyn)

    # Load initial configuration from file or dict
    # Supports both file paths (XYZ format) and explicit dict with species/coordinates
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
    # When batch_size > 1, coordinates/velocities/accelerations are shape (batch_size, N, 3)
    # Single trajectory per batch_size element; all run in parallel with JAX vectorization
    batch_size = general_params.get("batch_size", 1)
    assert batch_size >= 1, "batch_size must be at least 1"

    # Batch-size guard for variance tracking
    # Model etot_ensemble_var is per-frame per-trajectory, not aggregatable across batch trajectories
    if track_variance and batch_size > 1:
        print(
            "# WARNING: track_variance is enabled but batch_size > 1. "
            "etot_ensemble_var is a per-frame per-trajectory quantity and cannot be "
            "meaningfully tracked across batch trajectories. Disabling track_variance."
        )
        track_variance = False

    # random rotation
    do_random_rotation = input_params.get("random_rotation", True)
    if do_random_rotation:
        coordinates = apply_random_rotation(coordinates, n_rotations=batch_size)
        if verbose:
            print("# Applied random rotation to initial configuration.")
    else:
        coordinates = np.tile(coordinates, (batch_size, 1, 1))  # (batch_size,N,3)

    batch_species = np.tile(species, batch_size)  # (batch_size*N)

    # Sample velocities from Maxwell-Boltzmann distribution at temperature
    # Each trajectory in batch gets independent velocity random sampling
    # Shape: (batch_size, N, 3) - velocities per atom per trajectory
    temperature = general_params.get("temperature", 0.0)  # Kelvin
    assert temperature >= 0.0, "Temperature must be non-negative"
    velocities = sample_velocities(batch_species, temperature).reshape(
        batch_size, -1, 3
    )  # (batch_size,N,3)
    if verbose:
        print(f"# Sampled velocities at T={temperature} K.")

    # Remove center of mass (COM) linear and angular velocities per trajectory
    # Critical for energy conservation: removes unwanted drift in lab frame
    # Must do this per-trajectory in batch (each has independent COM)
    for b in range(batch_size):
        velocities[b] = remove_com_velocity(coordinates[b], velocities[b], species)
    if verbose:
        print("# Removed center of mass linear and angular velocities.")

    # coordinates bounding box
    molecule_radius = np.max(np.linalg.norm(coordinates, axis=-1))  # angstrom

    if collision:
        # Initialize projectile(s) for collision dynamics
        # Sample batch_size projectiles at random impact parameters, approaching target
        projectile_species = int(projectile_params.get("projectile_species", 18))  # default Argon
        max_impact_parameter = float(projectile_params.get("max_impact_parameter", 0.5))  # angstrom
        projectile_distance = float(projectile_params.get(
            "projectile_distance", 10.0 + 2 * molecule_radius
        ))  # angstrom
        assert (
            projectile_distance > 2 * molecule_radius
        ), f"projectile_distance must be larger than twice molecule radius ({2*molecule_radius:.2f} A)"

        # Projectile temperature: thermal sampling for incident velocity distribution
        projectile_temperature = float(projectile_params.get(
            "projectile_temperature", temperature
        ))  # Kelvin
        assert projectile_temperature > 0.0, "projectile_temperature must > 0 K"
        # Sample batch_size independent projectile trajectories
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
    # FeNNol model is the neural network potential for energy/force computation
    model_file = calculation_params.get("model") or calculation_params.get("model_file")
    if model_file is None:
        model_file = simulation_parameters.get("model") or simulation_parameters.get("model_file")
    if model_file is None:
        raise ValueError("Configuration must contain either 'model_file' or 'model' key")
    
    model_path = Path(model_file)
    assert model_path.is_file(), f"Model file {model_file} does not exist."
    print(f"# Using FENNIX model from file: {model_file}")
    
    # Load FeNNol model (FENNOL_MODULES_PATH already set in __init__.py)
    # Model provides energy and forces via JAX-compatible functions
    from fennol import FENNIX
    model = FENNIX.load(model_file)
    model.preproc_state = model.preproc_state.copy({"check_input": False})
    # Energy unit conversion: model outputs in its native units, convert to kcal/mol
    energy_conv = 1.0 / us.get_multiplier(model.energy_unit)

    initial_conformation = format_batch_conformations(
        species, coordinates, total_charge=total_charge
    )

    if collision:
        # Setup ZBL repulsion potential for projectile-target interactions
        # Repulsion computed separately from ML potential and added to total forces
        repulsion_energies_and_forces = setup_repulsion_potential(
            species, projectile_species, use_jax=True
        )

        def total_energies_and_forces(full_coordinates, conformation):
            # Collision mode: full_coordinates are (batch_size, N+1, 3) with projectile at index 0
            coordinates_model = full_coordinates[:, 1:, :]  # Extract target atoms (skip projectile)
            projectile_coordinates = full_coordinates[:, 0, :]  # Extract projectile position(s)
            
            # Compute ML potential energy and forces from model
            energies_model, forces_model, aux = model._energy_and_forces(model.variables, conformation)
            
            # Extract per-frame ensemble variance if model provides it (units: eV^2)
            # Always return a JAX array so JIT tracing is not broken when key is absent
            if isinstance(aux, dict) and "etot_ensemble_var" in aux:
                etot_ensemble_var = aux["etot_ensemble_var"]
            else:
                # Use NaN as sentinel for "missing" ensemble variance
                etot_ensemble_var = jnp.full_like(energies_model, jnp.nan)
            
            # Compute repulsion energy and forces for projectile-target interactions
            energies_repulsion, forces_repulsion, projectile_forces = (
                repulsion_energies_and_forces(coordinates_model, projectile_coordinates)
            )
            # Total energy: ML potential + repulsion potential (ML converted to kcal/mol)
            total_energies = energies_model * energy_conv + energies_repulsion
            # Total forces: ML forces + repulsion forces
            total_forces = forces_model.reshape(coordinates_model.shape[0], -1, 3) * energy_conv + forces_repulsion

            # Assemble full force array: projectile forces + target forces
            full_forces = jnp.concatenate(
                [projectile_forces[:, None, :], total_forces], axis=1
            )  # (batch_size,N+1,3)
            return total_energies, full_forces, etot_ensemble_var

        # Build full species array: [projectile, ...target_atoms]
        full_species = np.concatenate(
            [np.array([projectile_species], dtype=np.int32), species]
        )  # (N+1,)
        # Build full coordinate arrays with projectile as first particle
        full_coordinates = np.concatenate(
            [projectile_coordinates[:, None, :], coordinates], axis=1
        )  # (batch_size,N+1,3)
        full_velocities = np.concatenate(
            [projectiles_velocities[:, None, :], velocities], axis=1
        )  # (batch_size,N+1,3)

    else:
        # Non-collision mode: molecule only (no projectile particle)
        def total_energies_and_forces(full_coordinates, conformation):
            # Non-collision mode: full_coordinates are (batch_size, N, 3)
            coordinates_model = full_coordinates[:, :, :]
            
            # Compute ML potential energy and forces from model
            energies_model, forces_model, aux = model._energy_and_forces(model.variables, conformation)
            
            # Extract per-frame ensemble variance if model provides it (units: eV^2).
            # Always return a JAX array sentinel when missing so JIT tracing remains stable.
            if isinstance(aux, dict) and "etot_ensemble_var" in aux:
                etot_ensemble_var = aux["etot_ensemble_var"]
            else:
                etot_ensemble_var = jnp.full_like(energies_model, jnp.nan)
            
            # Total energy and forces (no repulsion in non-collision mode)
            total_energies = energies_model * energy_conv
            total_forces = forces_model.reshape(coordinates_model.shape[0], -1, 3) * energy_conv
            full_forces = total_forces  # (batch_size,N,3)
            return total_energies, full_forces, etot_ensemble_var

        full_species = species.copy()  # (N,)
        full_coordinates = coordinates.copy()  # (batch_size,N,3)
        full_velocities = velocities.copy()  # (batch_size,N,3)

    # Compute initial accelerations: a = F / m
    # Convert atomic masses from AMU to atomic units (DA / us.DA)
    # Masses shape: (N,) → reshape to (1, N, 1) for batch broadcasting
    masses_np = ATOMIC_MASSES[full_species].astype(np.float32) / us.DA  # (N,) in atomic units
    masses = jnp.array(masses_np)[None, :, None]  # (1,N,1) for broadcasting in jax

    # Preprocess initial conformation for the model (batches normalizations, padding, etc.)
    conformation = model.preprocess(use_gpu=True, **initial_conformation)
    # Ensure full_coordinates passed as jnp arrays to energy/force function
    full_coordinates_jnp = jnp.array(full_coordinates, dtype=jnp.float32)
    energies, forces, _ = total_energies_and_forces(full_coordinates_jnp, conformation)
    accelerations = forces / masses  # (batch_size, N(+1), 3) depending on collision

    # prepare integrator
    # dt_dyn is given in fs in the input, convert to ps (internal unit)
    # All energies/forces are in kcal/mol; time is in picoseconds
    dt_fs = dynamic_params.get("dt_dyn", dynamic_params.get("dt", 1.0))  # fs
    dt = dt_fs / 1000.0  # Convert fs to ps
    dt2 = dt * 0.5  # Half timestep for velocity Verlet

    @jax.jit
    def integrate_part1(coordinates, velocities, accelerations):
        # Velocity Verlet integration - part 1: update velocities and coordinates
        # v(t+dt/2) = v(t) + a(t) * dt/2
        # x(t+dt) = x(t) + v(t+dt/2) * dt
        velocities = velocities + accelerations * dt2  # (batch_size,N+1,3)
        coordinates = coordinates + velocities * dt  # (batch_size,N+1,3)
        return coordinates, velocities
    
    @jax.jit
    def integrate_part2(coordinates, velocities, conformation):
        # Velocity Verlet integration - part 2: compute new accelerations and update velocities
        # a(t+dt) computed from forces at new coordinates
        # v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2
        energies, forces, _var = total_energies_and_forces(coordinates, conformation)
        accelerations = forces / masses  # (batch_size,N+1,3)
        velocities = velocities + accelerations * dt2  # (batch_size,N,3)
        return velocities, accelerations, energies  # energies is (batch_size,)

    def integrate(initial_coordinates, initial_velocities, accelerations, step=0, energy_output_file=None, energy_steps=100):
        # Main dynamics integration step using Velocity Verlet algorithm
        # Called once per timestep by main loop
        # Returns updated coordinates, velocities, accelerations, energies, and optional energy output data
        
        # Part 1: Update velocities and coordinates with current accelerations
        coordinates, velocities = integrate_part1(
            initial_coordinates, initial_velocities, accelerations
        )

        # Update conformation for model preprocessing (normalize/pad atomic coordinates)
        if collision:
            # For collision dynamics, exclude projectile (index 0) from model input
            coords_for_model = coordinates[:,1:,:]
        else:
            # For non-collision dynamics, use all coordinates
            coords_for_model = coordinates

        # Preprocess coordinates for model (handles batching, normalization, etc.)
        conformation = model.preprocess(use_gpu=True, **update_conformation(
            initial_conformation, coords_for_model))

        # Part 2: Compute new accelerations at updated positions and complete velocity update
        velocities, accelerations, energies = integrate_part2(
            coordinates, velocities, conformation
        )

        # Extract per-frame ensemble variance from model (when track_variance enabled)
        # Done outside JIT to allow per-step access even if energy file written less frequently
        frame_variance = None
        if track_variance:
            try:
                _, _, aux_var = total_energies_and_forces(coordinates, conformation)
                if aux_var is not None:
                    frame_variance = np.atleast_1d(np.array(aux_var))
            except Exception:
                frame_variance = None

        # Write energy output file if requested (every energy_steps steps)
        energy_data = None
        if energy_output_file is not None and step % energy_steps == 0:
            energy_data = write_energy_output(
                energy_output_file, step, velocities, masses, energies, dt,
                frame_variance=frame_variance
            )

        return coordinates, velocities, accelerations, energies, energy_data, frame_variance

    def write_energy_output(output_file, step, velocities, masses, potential_energies, dt, frame_variance=None):
        """
        Write energy data to output file(s) for each trajectory in batch.
        
        Computes kinetic energy from velocities, combines with potential energy,
        calculates temperature, and writes to file(s) in FeNNol format.
        Per-trajectory energy files for batch_size > 1 (e.g., energies_0.out, energies_1.out).
        """
        # Compute kinetic energy per trajectory: 0.5 * m * v^2 summed over all atoms
        # masses is shaped (1, N, 1) for broadcasting; velocities is (batch_size, N, 3)
        kinetic_energies = 0.5 * jnp.sum(masses * velocities**2, axis=(1, 2))
        
        # Ensure all energy arrays are 1D: (batch_size,)
        potential_energies = jnp.squeeze(potential_energies)
        kinetic_energies = jnp.squeeze(kinetic_energies)
        
        # Total energy per trajectory
        total_energies = potential_energies + kinetic_energies
        
        # Calculate temperature from kinetic energy
        # T = (2 * Ekin) / (3 * N_atoms * k_B)  where k_B in kcal/(mol*K)
        k_B = 0.0019872043  # Boltzmann constant in kcal/(mol*K)
        N_atoms = velocities.shape[1]  # Number of particles (N or N+1 if collision)
        temperatures = (2.0 * kinetic_energies) / (3.0 * N_atoms * k_B)
        
        # Convert time to femtoseconds and determine output precision based on dt
        time_fs = step * dt * 1000.0  # dt is in ps, convert to fs
        dt_fs = dt * 1000.0  # Convert dt from ps to fs
        # If dt >= 1 fs, output no decimals; otherwise output appropriate decimals
        if dt_fs >= 1.0:
            time_decimals = 0
        else:
            time_decimals = len(str(dt_fs).split('.')[-1].rstrip('0'))
        time_format = f"{{:.{time_decimals}f}}"
        
        # Convert JAX arrays to numpy for file I/O
        total_energies_np = np.atleast_1d(np.array(total_energies))
        potential_energies_np = np.atleast_1d(np.array(potential_energies))
        kinetic_energies_np = np.atleast_1d(np.array(kinetic_energies))
        temperatures_np = np.atleast_1d(np.array(temperatures))

        # Normalize frame_variance to 1D numpy array of length batch_size, or None
        if frame_variance is not None:
            frame_variance = np.atleast_1d(np.array(frame_variance))
        
        # Write energy data to file(s): one per trajectory if batch_size > 1
        for b in range(batch_size):
            # Determine output file path for trajectory b
            if isinstance(output_file, list):
                # User provided explicit list of filenames (one per trajectory)
                file_path = output_file[b]
            else:
                # Single filename: use as-is if batch_size==1, else append _b suffix
                file_path = output_file if batch_size == 1 else f"{output_file}_{b}"

            # Create file with header if step == 0 (first write)
            if step == 0:
                with open(file_path, 'w') as f:
                    # Header line with column names
                    header = f"{'Step':>8s} {'Time[fs]':>12s} {'Etot':>12s} {'Epot':>12s} {'Ekin':>12s} {'Temp[K]':>10s}"
                    if track_variance:
                        # Add variance column if tracking model per-frame ensemble variance
                        header += f" {'Var(eV^2)':>12s}"
                    header += "\n"
                    f.write(header)

            # Append energy data row(s)
            with open(file_path, 'a') as f:
                # Extract scalar values for trajectory b (handle both 1D and multi-element arrays)
                total_e = float(total_energies_np.flat[b] if total_energies_np.size > 1 else total_energies_np)
                pot_e = float(potential_energies_np.flat[b] if potential_energies_np.size > 1 else potential_energies_np)
                kin_e = float(kinetic_energies_np.flat[b] if kinetic_energies_np.size > 1 else kinetic_energies_np)
                temp = float(temperatures_np.flat[b] if temperatures_np.size > 1 else temperatures_np)

                # Format time with appropriate precision
                time_str = time_format.format(time_fs)
                # Build output line: step | time | etot | epot | ekin | temp
                line = f"{step:8d} {time_str:>12s} {total_e:12.6f} {pot_e:12.6f} {kin_e:12.6f} {temp:10.2f}"
                if track_variance:
                    # Include model ensemble variance (eV^2) or "None" if not available
                    if frame_variance is not None:
                        var_val = float(frame_variance.flat[b] if frame_variance.size > 1 else frame_variance[0])
                        line += f" {var_val:12.5f}\n"
                    else:
                        line += f" {'None':>12s}\n"
                else:
                    line += "\n"
                f.write(line)

        # Return aggregated energy data for summary statistics (accumulated in main loop)
        # Variances from model (units: eV^2); None if not provided by model
        var_arr = frame_variance  # already numpy or None

        return {
            'total_energies': total_energies_np,
            'potential_energies': potential_energies_np,
            'kinetic_energies': kinetic_energies_np,
            'temperatures': temperatures_np,
            'variances': var_arr,
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
