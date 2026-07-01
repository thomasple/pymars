#!/usr/bin/env python3
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from fennol.models import FENNIX
from fennol.utils.io import last_xyz_frame
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX
from fennol.utils.atomic_units import au
from fennol.md.utils import optimize_fire2

from pymars.utils import write_xyz_frame, format_batch_conformations, us

def run_opt(xyz_file, model_file, outfile=None, dt=0.002,
            total_charge=0, tolerance=1e-2, max_steps=10000, keep_every=-1, dxmax=0.2):
    "Minimize the energy of a molecular geometry using a FENNIX model"

    # read the coordinates from the xyz file
    print(f"Reading coordinates from: {xyz_file}")
    symbols, coordinates, comment = last_xyz_frame(
        xyz_file)
    print(f"Read {len(symbols)} atoms.")
    coordinates = np.array(coordinates)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)
    nat = len(species)
    if total_charge != 0:
        print(f"Using total charge of {total_charge} for the system.")
    inputs = {
        "species": species,
        "natoms": np.array([nat], dtype=np.int32),
        "batch_index": np.array([0] * nat, dtype=np.int32),
        "total_charge": np.array([total_charge], dtype=np.int32),
    }

    xyz_filepath = Path(xyz_file)

    # Load the FENNIX model
    model_file = Path(model_file)
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=False)
    convert = au.KCALPERMOL / model.Ha_to_model_energy 

    def energy_force_fn(coordinates):
        e, f, _ = model.energy_and_forces(
            **inputs, coordinates=coordinates, gpu_preprocessing=True
        )
        e = float(e[0]) * convert / nat
        f = np.array(f) * convert
        return e, f

    # Optimize the geometry using FIRE algorithm
    print("Starting geometry optimization...")
    results = optimize_fire2(
        coordinates,
        energy_force_fn,
        atol=tolerance,
        dt=dt,
        Nmax=max_steps,
        logoutput=True,
        keep_every=keep_every,
        max_disp=dxmax,  # convert Angstroms to Bohr
    )
    coordinates = results[0]
    success = results[1]
    print("#######################################################")
    if success:
        print("Optimization converged successfully!")
    else:
        print("Optimization did not converge... Writing the last frame anyway.")


    #If selected, write the trajectory of the optimization to a file
    if keep_every > 0:
        traj_file = xyz_filepath.with_suffix(".trj.xyz")
        with open(traj_file, "w") as f:
            for step, coords in enumerate(results[2]):
                if step % keep_every == 0:
                    write_xyz_frame(f, symbols, coords, comment=f"Step {step}")
        print(f"Optimization trajectory written to {traj_file}")

    print("#######################################################")
    print(f"Preparing final single-point energy calculation for the optimized geometry...")

    model.preproc_state = model.preproc_state.copy({"check_input": False})
    # Energy unit conversion: model outputs in its native units, convert to kcal/mol
    energy_conv = 1.0 / us.get_multiplier(model.energy_unit)

    # Preprocess initial conformation for the model (batches normalizations, padding, etc.)
    coordinates_jnp = jnp.array(coordinates)[None, :, :]  # Add batch dimension
    pre_conformation = format_batch_conformations(species, coordinates_jnp, total_charge)
    conformation = model.preprocess(
    use_gpu=True,
    **pre_conformation
    )

    def total_energies_and_forces(full_coordinates, conformation):
        coordinates_model = full_coordinates[None, :, :]

        # Compute ML potential energy and forces from model
        energies_model, forces_model, aux = model._energy_and_forces(model.variables, conformation)
        
        #print(f"DEBUG: model provided energies with shape {energies_model.shape}, energies sample: {energies_model}")
        #print(f"DEBUG: model provided forces with shape {forces_model.shape}, forces sample: {forces_model}")
        #Extract  ensemble charges if model provides them (units: e). 
        # Always return a JAX array sentinel when missing so JIT tracing remains stable.
        if isinstance(aux, dict) and "charges" in aux:
            #print(f"DEBUG: model provided charges with shape {aux['charges'].shape}, charges sample: {aux['charges']}")# (batch_size * N,)
            full_charges = aux["charges"].reshape(
                full_coordinates.shape[0],
                full_coordinates.shape[1]
                ) # (batch_size, N)
            #print(f"DEBUG:  charges have shape {full_charges.shape}, charges sample: {full_charges}")
        else:
            full_charges = None
        
        # Total energy and forces
        total_energies = energies_model * energy_conv
        total_forces = forces_model.reshape(coordinates_model.shape[0], -1, 3) * energy_conv
        full_forces = total_forces  # (batch_size,N,3)
        return total_energies, full_forces, full_charges

    # Calculate the energy, forces and partial charges for the geometry
    energy, forces, partial_charges = total_energies_and_forces(coordinates_jnp, conformation)
    print("\n")
    print("#######################################################")
    print(f"Final singlepoint energy: {(energy[0]):.6f} kcal/mol")
    print("\n")
    print(f"Forces on atoms (kcal/mol/Angstrom):")
    print(f"Index\tAtom\tFx\t\tFy\t\tFz")
    for i, (s, f) in enumerate(zip(symbols, forces[0])):
        print(f"{i+1}\t{s}\t{f[0]:.6f}\t{f[1]:.6f}\t{f[2]:.6f}")
    print("\n")
    if partial_charges is not None:
        print("Mulliken partial charges:")
        print(f"Index\tAtom\tCharge")
        for i, (s, q) in enumerate(zip(symbols, partial_charges[0])):
            print(f"{i+1}\t{s}\t{q:.6f}")
    else:
        print("No partial charges were computed by the model.")

    # write the output
    output_file = outfile if outfile else xyz_filepath.with_suffix(".opt.xyz")
    with open(output_file, "w") as f:
        write_xyz_frame(f, symbols, coordinates, partial_charges, comment=f"Geometry optimized with FENNIX model: {model_file.name}")
    print("\n")
    print(f"Final configuration written to {output_file}:")
    print(f"Number of atoms: {len(symbols)}")
    with open(output_file, "r") as f:
        lines = f.readlines()
    # Skip atom count (line 0) and comment (line 1)
    for line in lines[2:]:
        print(line.rstrip())
    print("#######################################################")
    print(f"Optimization complete. Optimized geometry written to {output_file}.")
    print("  Geometry optimization completed successfully!")
    print("#######################################################")

