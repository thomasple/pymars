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
  - integrate(...) function to run dynamics (see docstring below)
  - batch_size, dt
"""
    
    # load type of dynamics (accept boolean or string)
    coll_dyn = simulation_parameters.get("collision_dynamics", True)
    if isinstance(coll_dyn, str):
        collision = coll_dyn.strip().upper() in ("TRUE", "YES", "1")
    else:
        collision = bool(coll_dyn)
    
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
    batch_size = int(simulation_parameters.get("batch_size", 1))
    assert batch_size >= 1, "batch_size must be at least 1"
    
    # random rotation (or tile)
    do_random_rotation = simulation_parameters.get("random_rotation", True)
    if do_random_rotation:
        coordinates = apply_random_rotation(coordinates, n_rotations=batch_size)
        if verbose:
            print("# Applied random rotation to initial configuration.")
    else:
        coordinates = np.tile(coordinates, (batch_size, 1, 1))  # (batch_size,N,3)
    
    batch_species = np.tile(species, batch_size)  # (batch_size*N)  -- may be unused later
    
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
    
    if collision:
        # initialize projectiles
        projectile_species = int(simulation_parameters.get("projectile_species", 18))  # default Argon
        max_impact_parameter = float(simulation_parameters.get("max_impact_parameter", 0.5))  # angstrom
        projectile_distance = float(simulation_parameters.get("projectile_distance", 10.0 + 2 * molecule_radius))  # angstrom
        assert (
            projectile_distance > 2 * molecule_radius
        ), f"projectile_distance must be larger than twice molecule radius ({2*molecule_radius:.2f} A)"
    
        projectile_temperature = float(simulation_parameters.get("projectile_temperature", temperature))  # Kelvin
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
    assert Path(model_file).is_file(), f"Model file {model_file} does not exist."
    print(f"# Using FENNIX model from file: {model_file}")
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
    # masses for jax computations and numpy return
    masses_np = ATOMIC_MASSES[full_species].astype(np.float32) / us.DA  # (N,) in atomic units
    masses = jnp.array(masses_np)[None, :, None]  # (1,N,1) for broadcasting in jax
    
    # Preprocess initial conformation for the model
    conformation = model.preprocess(use_gpu=True, **initial_conformation)
    # Ensure full_coordinates passed as jnp arrays to energy/force function
    full_coordinates_jnp = jnp.array(full_coordinates, dtype=jnp.float32)
    energies, forces = total_energies_and_forces(full_coordinates_jnp, conformation)
    accelerations = forces / masses  # (batch_size, N(+1), 3) depending on collision
    
    # prepare integrator
    dt = simulation_parameters.get("dt", 1.0 / us.FS)
    dt2 = dt * 0.5
    
    @jax.jit
    def integrate_part1(coordinates, velocities, accelerations):
        # Velocity Verlet step - part 1
        velocities = velocities + accelerations * dt2
        coordinates = coordinates + velocities * dt
        return coordinates, velocities
    
    @jax.jit
    def integrate_part2(coordinates, velocities, conformation):
        energies, forces = total_energies_and_forces(coordinates, conformation)
        accelerations = forces / masses
        velocities = velocities + accelerations * dt2
        return velocities, accelerations, energies
    
    def integrate(initial_coordinates, initial_velocities, accelerations,
                  nsteps=10000, fragmentation_check=False, distance_threshold=10.0, 
                  cell=None, verbose=True):
        """
        Run dynamics for up to `nsteps` steps using the Velocity Verlet integrator.
    
        Args:
            fragmentation_check: If True, monitor fragmentation and stop when fragments separate
            distance_threshold: Minimum COM distance (Angstrom) between fragments to consider "separated"
            cell: Optional 3x3 array for PBC in topology detection
            verbose: Print progress information
    
        Behavior for fragmentation_check with batch_size > 1:
          - The integrator runs all batch members in parallel.
          - We monitor topology per-batch (ignoring projectile index when collision=True).
          - For each batch member we record its starting number of graphs N and set a
            target of N+1. A batch member is considered "split" when its graph count
            reaches >= N+1 AND the fragments are separated by >= distance_threshold.
          - The integration continues until all batch members have fragmented and separated
            or until nsteps is reached.
        """
        coords = initial_coordinates  # jnp array (batch_size, N(+1), 3) or (batch_size, N, 3)
        vels = initial_velocities
        accs = accelerations
    
        bs = int(coords.shape[0])
    
        # bookkeeping for split detection
        active = np.ones(bs, dtype=bool)
        split_steps = -np.ones(bs, dtype=int)  # -1 means not split yet; -2 will mark already >= target before integration (rare)
        initial_graphs = np.zeros(bs, dtype=int)
    
        # Determine initial graph counts (host copy)
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
    
        # target for each batch member is initial_graphs + 1
        target_graphs = initial_graphs + 1
    
        # If any member already meets or exceeds target (unlikely), mark as already satisfied
        for b in range(bs):
            if initial_graphs[b] >= target_graphs[b]:
                active[b] = False
                split_steps[b] = -2
    
        if verbose:
            print(f"# initial graph counts per batch: {initial_graphs.tolist()}")
            print(f"# target graph counts per batch (N+1): {target_graphs.tolist()}")
            print(f"# running up to {nsteps} steps; fragmentation_check={fragmentation_check}, distance_threshold={distance_threshold} A")
    
        final_energies = None
        fragment_distances = -np.ones(bs, dtype=float)  # Track fragment separations
    
        for step in range(nsteps):
            # integrate one step for all batch members (JAX jitted)
            coords, vels = integrate_part1(coords, vels, accs)
    
            # update conformation for model preprocessing using molecule coordinates (skip projectile index)
            coords_for_prep = coords[:, 1:, :] if collision else coords
            conformation = model.preprocess(use_gpu=True, **update_conformation(initial_conformation, coords_for_prep))
    
            vels, accs, energies = integrate_part2(coords, vels, conformation)
            final_energies = energies
    
            if fragmentation_check:
                coords_host = jax.device_get(coords)  # move to host for topology detection
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
                    
                    # Check if fragmented
                    if ngraphs >= target_graphs[b]:
                        # Compute fragment separation distance
                        frag_dist = get_fragment_separation(species_mol, mol_coords, masses_mol, cell)
                        fragment_distances[b] = frag_dist
                        
                        # Mark as split when fragments are separated beyond threshold
                        if frag_dist >= distance_threshold:
                            active[b] = False
                            split_steps[b] = step
                            if verbose:
                                print(f"# batch {b} fragmented and separated at step {step} "
                                      f"(initial {initial_graphs[b]} -> now {ngraphs} graphs, "
                                      f"fragment distance: {frag_dist:.2f} A >= {distance_threshold} A)")
                        elif verbose and step % max(1, nsteps // 20) == 0:
                            print(f"# batch {b} fragmented but not yet separated "
                                  f"({ngraphs} graphs, fragment distance: {frag_dist:.2f} A < {distance_threshold} A)")
    
            # stop when all batch members have fragmented and separated
            if fragmentation_check and not active.any():
                if verbose:
                    print(f"# all batch members reached their N+1 target by step {step}. Stopping integration.")
                break
    
            # optional progress logging
            if verbose and (step % max(1, nsteps // 10) == 0):
                n_active = int(active.sum())
                print(f"# step {step:6d}: active simulations remaining: {n_active}")
    
        # prepare return values (keep arrays as jax arrays for downstream jax use; provide split info as numpy)
        return {
            "coordinates": coords,
            "velocities": vels,
            "accelerations": accs,
            "energies": final_energies,
            "split_steps": split_steps,  # -2: already >= target at start, -1: never reached target, >=0: step when fragmented and separated
            "fragment_distances": fragment_distances,  # -1: not fragmented, >=0: minimum COM distance between fragments (A)
            "all_reached_target": np.all(split_steps != -1),
            "initial_graphs": initial_graphs,
            "target_graphs": target_graphs,
        }
    
    # return context
    return {
        "species": full_species,
        "masses": masses_np.reshape(-1),  # numpy 1D array of masses
        "coordinates": jnp.array(full_coordinates, dtype=jnp.float32),
        "velocities": jnp.array(full_velocities, dtype=jnp.float32),
        "accelerations": jnp.array(accelerations, dtype=jnp.float32),
        "total_energies_and_forces": total_energies_and_forces,
        "integrate": integrate,
        "batch_size": batch_size,
        "dt": dt,
    }