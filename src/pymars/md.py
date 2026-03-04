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

from .topology import count_graphs, get_fragment_separation


def initialize_collision_simulation(simulation_parameters, verbose=True):
    """
    Initialize collision (or molecule-only) simulation.

    Returns a dict with:
      - species, masses (numpy), coordinates (jnp), velocities (jnp), accelerations (jnp)
      - total_energies_and_forces callable
      - integrate(...) function to run dynamics (simple one-step, used by __init__.py)
      - integrate_fragmentation(...) function for topology-aware multi-step dynamics
      - batch_size, dt
    """

    # Extract nested parameter sections
    input_params = simulation_parameters.get("input_parameters", {})
    general_params = simulation_parameters.get("general_parameters", {})
    projectile_params = simulation_parameters.get("projectile_parameters", {})
    calculation_params = simulation_parameters.get("calculation_parameters", {})
    dynamic_params = simulation_parameters.get("dynamic_parameters", {})
    # Output options
    output_params = simulation_parameters.get("output_details", {})
    track_variance = bool(output_params.get("track_variance", False))
    # Variance tracking requires batch_size==1 (model etot_ensemble_var is per-frame,
    # not across trajectories).  The guard below will be enforced after batch_size is known.

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
        species, coordinates = read_initial_configuration(geometry)
    elif isinstance(geometry, dict):
        species = np.array(geometry["species"], dtype=np.int32)
        coordinates = np.array(geometry["coordinates"], dtype=np.float32).reshape(-1, 3)
    else:
        raise ValueError("initial_geometry must be a file path or a dictionary")

    total_charge = round(input_params.get("total_charge", simulation_parameters.get("total_charge", 0)))
    if verbose:
        print(f"# total charge of the system: {total_charge} e")

    if verbose:
        composition_str = get_composition_string(species)
        print("# Initial configuration loaded: ", composition_str)

    # batch size for parallel simulations
    batch_size = int(general_params.get("batch_size", simulation_parameters.get("batch_size", 1)))
    assert batch_size >= 1, "batch_size must be at least 1"

    # Batch-size guard for variance tracking
    if track_variance and batch_size > 1:
        print(
            "# WARNING: track_variance is enabled but batch_size > 1. "
            "etot_ensemble_var is a per-frame per-trajectory quantity and cannot be "
            "meaningfully tracked across batch trajectories. Disabling track_variance."
        )
        track_variance = False

    # random rotation (or tile)
    do_random_rotation = input_params.get("random_rotation", simulation_parameters.get("random_rotation", True))
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
        projectile_species = int(projectile_params.get(
            "projectile_species", simulation_parameters.get("projectile_species", 18)))
        max_impact_parameter = float(projectile_params.get(
            "max_impact_parameter", simulation_parameters.get("max_impact_parameter", 0.5)))
        projectile_distance = float(projectile_params.get(
            "projectile_distance", simulation_parameters.get("projectile_distance", 10.0 + 2 * molecule_radius)))
        assert (
            projectile_distance > 2 * molecule_radius
        ), f"projectile_distance must be larger than twice molecule radius ({2*molecule_radius:.2f} A)"

        projectile_temperature = float(projectile_params.get(
            "projectile_temperature", simulation_parameters.get("projectile_temperature", temperature)))
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
                f"# initialized projectile at distance {projectile_distance:.2f} A "
                f"with temperature {projectile_temperature} K"
            )
            print(f"# Time before collision: ~{time_to_impact:.2f} ps")

    # Support both "model" and "model_file" keys for backward compatibility
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
            coordinates_model = full_coordinates[:, 1:, :]
            projectile_coordinates = full_coordinates[:, 0, :]
            energies_model, forces_model, aux = model._energy_and_forces(
                model.variables, conformation)
            # Extract per-frame ensemble variance if model provides it (eV^2).
            if isinstance(aux, dict) and "etot_ensemble_var" in aux:
                etot_ensemble_var = aux["etot_ensemble_var"]
            else:
                etot_ensemble_var = jnp.full_like(energies_model, jnp.nan)
            energies_repulsion, forces_repulsion, projectile_forces = (
                repulsion_energies_and_forces(coordinates_model, projectile_coordinates)
            )
            total_energies = energies_model * energy_conv + energies_repulsion
            total_forces = (forces_model.reshape(coordinates_model.shape[0], -1, 3)
                            * energy_conv + forces_repulsion)
            full_forces = jnp.concatenate(
                [projectile_forces[:, None, :], total_forces], axis=1
            )
            return total_energies, full_forces, etot_ensemble_var

        full_species = np.concatenate(
            [np.array([projectile_species], dtype=np.int32), species]
        )
        full_coordinates = np.concatenate(
            [projectile_coordinates[:, None, :], coordinates], axis=1
        )
        full_velocities = np.concatenate(
            [projectiles_velocities[:, None, :], velocities], axis=1
        )

    else:
        def total_energies_and_forces(full_coordinates, conformation):
            coordinates_model = full_coordinates[:, :, :]
            energies_model, forces_model, aux = model._energy_and_forces(
                model.variables, conformation)
            etot_ensemble_var = (aux.get("etot_ensemble_var", None)
                                 if isinstance(aux, dict) else None)
            total_energies = energies_model * energy_conv
            total_forces = (forces_model.reshape(coordinates_model.shape[0], -1, 3)
                            * energy_conv)
            return total_energies, total_forces, etot_ensemble_var

        full_species = species.copy()
        full_coordinates = coordinates.copy()
        full_velocities = velocities.copy()

    # compute initial accelerations
    masses_np = ATOMIC_MASSES[full_species].astype(np.float32) / us.DA
    masses = jnp.array(masses_np)[None, :, None]

    conformation = model.preprocess(use_gpu=True, **initial_conformation)
    full_coordinates_jnp = jnp.array(full_coordinates, dtype=jnp.float32)
    energies, forces, _ = total_energies_and_forces(full_coordinates_jnp, conformation)
    accelerations = forces / masses

    # prepare integrator
    dt_fs = dynamic_params.get("dt_dyn", dynamic_params.get("dt", 1.0))
    dt = dt_fs / 1000.0
    dt2 = dt * 0.5

    @jax.jit
    def integrate_part1(coordinates, velocities, accelerations):
        velocities = velocities + accelerations * dt2
        coordinates = coordinates + velocities * dt
        return coordinates, velocities

    @jax.jit
    def integrate_part2(coordinates, velocities, conformation):
        energies, forces, _var = total_energies_and_forces(coordinates, conformation)
        accelerations = forces / masses
        velocities = velocities + accelerations * dt2
        return velocities, accelerations, energies

    # ------------------------------------------------------------------ #
    #  integrate() -- simple one-step integrator (used by __init__.py loop)
    # ------------------------------------------------------------------ #
    def integrate(initial_coordinates, initial_velocities, accelerations,
                  step=0, energy_output_file=None, energy_steps=100):
        """One Velocity-Verlet step with optional energy output."""
        coordinates, velocities = integrate_part1(
            initial_coordinates, initial_velocities, accelerations
        )

        if collision:
            coords_for_model = coordinates[:, 1:, :]
        else:
            coords_for_model = coordinates

        conformation = model.preprocess(use_gpu=True, **update_conformation(
            initial_conformation, coords_for_model))

        velocities, accelerations, energies = integrate_part2(
            coordinates, velocities, conformation
        )

        # Extract per-frame variance from model (outside jit) when requested.
        frame_variance = None
        if track_variance:
            try:
                _, _, aux_var = total_energies_and_forces(coordinates, conformation)
                if aux_var is not None:
                    frame_variance = np.atleast_1d(np.array(aux_var))
            except Exception:
                frame_variance = None

        # Write energy output if requested
        energy_data = None
        if energy_output_file is not None and step % energy_steps == 0:
            energy_data = write_energy_output(
                energy_output_file, step, velocities, masses, energies, dt,
                frame_variance=frame_variance
            )

        return coordinates, velocities, accelerations, energies, energy_data, frame_variance

    # ------------------------------------------------------------------ #
    #  integrate_fragmentation() -- topology-aware multi-step integrator
    # ------------------------------------------------------------------ #
    def integrate_fragmentation(initial_coordinates, initial_velocities, accelerations,
                                nsteps=10000, fragmentation_check=False,
                                distance_threshold=10.0, cell=None, verbose=True):
        """
        Run dynamics for up to *nsteps* steps using Velocity Verlet.

        Args:
            fragmentation_check: If True, monitor fragmentation and stop when
                fragments separate.
            distance_threshold: Min COM distance (A) between fragments to
                consider "separated".
            cell: Optional 3x3 array for PBC in topology detection.
            verbose: Print progress information.

        Behavior for fragmentation_check with batch_size > 1:
          - All batch members run in parallel.
          - Topology is monitored per-batch (ignoring projectile when collision).
          - Target is initial_graphs + 1 per batch member.
          - Integration continues until all fragmented+separated or nsteps.
        """
        coords = initial_coordinates
        vels = initial_velocities
        accs = accelerations

        bs = int(coords.shape[0])

        active = np.ones(bs, dtype=bool)
        split_steps = -np.ones(bs, dtype=int)
        initial_graphs = np.zeros(bs, dtype=int)

        coords_host0 = jax.device_get(coords)
        for b in range(bs):
            if collision:
                mol_coords = coords_host0[b, 1:, :]
                species_mol = full_species[1:]
            else:
                mol_coords = coords_host0[b, :, :]
                species_mol = full_species
            ngraphs = count_graphs(species_mol, mol_coords, cell)
            initial_graphs[b] = ngraphs

        target_graphs = initial_graphs + 1

        for b in range(bs):
            if initial_graphs[b] >= target_graphs[b]:
                active[b] = False
                split_steps[b] = -2

        if verbose:
            print(f"# initial graph counts per batch: {initial_graphs.tolist()}")
            print(f"# target graph counts per batch (N+1): {target_graphs.tolist()}")
            print(f"# running up to {nsteps} steps; fragmentation_check={fragmentation_check}, "
                  f"distance_threshold={distance_threshold} A")

        final_energies = None
        fragment_distances = -np.ones(bs, dtype=float)

        for step in range(nsteps):
            coords, vels = integrate_part1(coords, vels, accs)

            coords_for_prep = coords[:, 1:, :] if collision else coords
            conformation = model.preprocess(use_gpu=True, **update_conformation(
                initial_conformation, coords_for_prep))

            vels, accs, energies = integrate_part2(coords, vels, conformation)
            final_energies = energies

            if fragmentation_check:
                coords_host = jax.device_get(coords)
                for b in range(bs):
                    if not active[b]:
                        continue
                    if collision:
                        mol_coords = coords_host[b, 1:, :]
                        species_mol = full_species[1:]
                        masses_mol = masses_np[1:]
                    else:
                        mol_coords = coords_host[b, :, :]
                        species_mol = full_species
                        masses_mol = masses_np

                    ngraphs = count_graphs(species_mol, mol_coords, cell)

                    if ngraphs >= target_graphs[b]:
                        frag_dist = get_fragment_separation(
                            species_mol, mol_coords, masses_mol, cell)
                        fragment_distances[b] = frag_dist

                        if frag_dist >= distance_threshold:
                            active[b] = False
                            split_steps[b] = step
                            if verbose:
                                print(
                                    f"# batch {b} fragmented and separated at step {step} "
                                    f"(initial {initial_graphs[b]} -> now {ngraphs} graphs, "
                                    f"fragment distance: {frag_dist:.2f} A >= {distance_threshold} A)")
                        elif verbose and step % max(1, nsteps // 20) == 0:
                            print(
                                f"# batch {b} fragmented but not yet separated "
                                f"({ngraphs} graphs, fragment distance: "
                                f"{frag_dist:.2f} A < {distance_threshold} A)")

            if fragmentation_check and not active.any():
                if verbose:
                    print(f"# all batch members reached their N+1 target by step {step}. "
                          "Stopping integration.")
                break

            if verbose and (step % max(1, nsteps // 10) == 0):
                n_active = int(active.sum())
                print(f"# step {step:6d}: active simulations remaining: {n_active}")

        return {
            "coordinates": coords,
            "velocities": vels,
            "accelerations": accs,
            "energies": final_energies,
            "split_steps": split_steps,
            "fragment_distances": fragment_distances,
            "all_reached_target": np.all(split_steps != -1),
            "initial_graphs": initial_graphs,
            "target_graphs": target_graphs,
        }

    # ------------------------------------------------------------------ #
    #  write_energy_output() -- energy file writer (from main)
    # ------------------------------------------------------------------ #
    def write_energy_output(output_file, step, velocities, masses,
                            potential_energies, dt, frame_variance=None):
        """Write energy data to output file in FeNNol format."""
        kinetic_energies = 0.5 * jnp.sum(masses * velocities**2, axis=(1, 2))

        potential_energies = jnp.squeeze(potential_energies)
        kinetic_energies = jnp.squeeze(kinetic_energies)

        total_energies = potential_energies + kinetic_energies

        k_B = 0.0019872043  # kcal/(mol*K)
        N_atoms = velocities.shape[1]
        temperatures = (2.0 * kinetic_energies) / (3.0 * N_atoms * k_B)

        time_fs = step * dt * 1000.0
        dt_fs_local = dt * 1000.0
        if dt_fs_local >= 1.0:
            time_decimals = 0
        else:
            time_decimals = len(str(dt_fs_local).split('.')[-1].rstrip('0'))
        time_format = f"{{:.{time_decimals}f}}"

        total_energies_np = np.atleast_1d(np.array(total_energies))
        potential_energies_np = np.atleast_1d(np.array(potential_energies))
        kinetic_energies_np = np.atleast_1d(np.array(kinetic_energies))
        temperatures_np = np.atleast_1d(np.array(temperatures))

        if frame_variance is not None:
            frame_variance = np.atleast_1d(np.array(frame_variance))

        for b in range(batch_size):
            if isinstance(output_file, list):
                file_path = output_file[b]
            else:
                file_path = output_file if batch_size == 1 else f"{output_file}_{b}"

            if step == 0:
                with open(file_path, 'w') as f:
                    header = (f"{'Step':>8s} {'Time[fs]':>12s} {'Etot':>12s} "
                              f"{'Epot':>12s} {'Ekin':>12s} {'Temp[K]':>10s}")
                    if track_variance:
                        header += f" {'Var(eV^2)':>12s}"
                    header += "\n"
                    f.write(header)

            with open(file_path, 'a') as f:
                total_e = float(total_energies_np.flat[b]
                                if total_energies_np.size > 1 else total_energies_np)
                pot_e = float(potential_energies_np.flat[b]
                              if potential_energies_np.size > 1 else potential_energies_np)
                kin_e = float(kinetic_energies_np.flat[b]
                              if kinetic_energies_np.size > 1 else kinetic_energies_np)
                temp = float(temperatures_np.flat[b]
                             if temperatures_np.size > 1 else temperatures_np)

                time_str = time_format.format(time_fs)
                line = (f"{step:8d} {time_str:>12s} {total_e:12.6f} "
                        f"{pot_e:12.6f} {kin_e:12.6f} {temp:10.2f}")
                if track_variance:
                    if frame_variance is not None:
                        var_val = float(frame_variance.flat[b]
                                        if frame_variance.size > 1
                                        else frame_variance[0])
                        line += f" {var_val:12.5f}\n"
                    else:
                        line += f" {'None':>12s}\n"
                else:
                    line += "\n"
                f.write(line)

        var_arr = frame_variance

        return {
            'total_energies': total_energies_np,
            'potential_energies': potential_energies_np,
            'kinetic_energies': kinetic_energies_np,
            'temperatures': temperatures_np,
            'variances': var_arr,
        }

    # ------------------------------------------------------------------ #
    #  Return context dictionary
    # ------------------------------------------------------------------ #
    return {
        "species": full_species,
        "masses": masses_np,
        "coordinates": jnp.array(full_coordinates, dtype=jnp.float32),
        "velocities": jnp.array(full_velocities, dtype=jnp.float32),
        "accelerations": jnp.array(accelerations, dtype=jnp.float32),
        "total_energies_and_forces": total_energies_and_forces,
        "integrate": integrate,
        "integrate_fragmentation": integrate_fragmentation,
        "batch_size": batch_size,
        "dt": dt,
        "initial_energies": energies,
    }
