import numpy as np
import jax.numpy as jnp
from pathlib import Path

from fennol.models import FENNIX
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX
from pymars.utils import format_batch_conformations, us

from pymars.phoboos.utils import read_xyz_file

def run_spt(xyz_file, model_file, outfile=None, total_charge=0):
    "Calculate the energy of a molecular geometry using a FENNIX model"
    # read the coordinates from the xyz file
    print(f"Reading coordinates from: {xyz_file}")
    symbols, coordinates, = read_xyz_file(xyz_file)
    print(f"Read {len(symbols)} atoms.")
    coordinates = np.array(coordinates)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)

    # Load the FENNIX model
    model_file = Path(model_file)
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=False)
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
    #print(conformation.keys())
    
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
    
    print("#######################################################")
    print(f"#  Single-point calculation completed successfully!   #")
    print("#######################################################")
