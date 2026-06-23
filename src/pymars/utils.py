from fennol.utils.atomic_units import UnitSystem
from fennol.utils.periodic_table import PERIODIC_TABLE
import numpy as np
import jax.numpy as jnp
import jax

__all__ = ['us','format_batch_conformations','update_conformation','get_composition_string']

us = UnitSystem(L="angstrom",T="ps",E="kcalpermol")

def format_batch_conformations(species,coordinates,total_charge=0):
    """Format batch conformations into a dictionary for FeNNol model processing.

    Args:
     - species: array of shape (Nat,) with atomic species indices
     - coordinates: array of shape (M,Nat,3) with atomic coordinates for M conformations
    Returns:
     - conformation: dict with keys 'species', 'coordinates','natoms','batch_index'
    """
    
    batch_size, Nat = coordinates.shape[0:2]
    species_batch = np.tile(species, batch_size)  # (M * Nat,)
    coordinates_batch = coordinates.reshape(-1, 3)  # (M * Nat, 3)
    batch_index = np.repeat(np.arange(batch_size,dtype=np.int32), Nat)  # (M * Nat,)
    natoms_batch = np.array([Nat] * batch_size,dtype=np.int32)  # list of length M
    use_x64 = bool(jax.config.read("jax_enable_x64"))
    coord_dtype = jnp.float64 if use_x64 else jnp.float32
    return {
        'species': jnp.array(species_batch, dtype=jnp.int32),
        'coordinates': jnp.array(coordinates_batch, dtype=coord_dtype),
        'natoms': jnp.array(natoms_batch, dtype=jnp.int32),
        'batch_index': jnp.array(batch_index, dtype=jnp.int32),
        'total_charge': jnp.array([total_charge] * batch_size, dtype=jnp.int32),
    }

def update_conformation(conformation, new_coordinates):
    """Update coordinates in a batch conformation dictionary.

    Args:
     - conformation: dict with keys 'species', 'coordinates','natoms','batch_index' 
        - new_coordinates: array of shape (M,Nat,3) with new atomic coordinates for M conformations 
    Returns:
     - updated_conformation: dict with updated 'coordinates'
    """
    use_x64 = bool(jax.config.read("jax_enable_x64"))
    coord_dtype = jnp.float64 if use_x64 else jnp.float32
    return {**conformation, 'coordinates': jnp.asarray(new_coordinates.reshape(-1,3),dtype=coord_dtype)}

def get_composition_string(species):
    """Get a string representation of the composition from species array.

    Args:
     - species: array of shape (N,) with atomic species indices
    Returns:
     - composition_str: string representing the composition, e.g. "O1_H2"    
    """
    species_set, counts = np.unique(species, return_counts=True)
    composition_str = "_".join(
        f"{PERIODIC_TABLE[int(s)]}{int(n)}" for s, n in zip(species_set, counts)
    )
    return composition_str


def write_xyz_frame(f, symbols, coordinates, charges=None, **kwargs):
    """Write a single XYZ frame.
    Supports an optional `comment` kwarg (2nd XYZ line) and optional per-atom charges.
    """
    nat = len(symbols)
    comment = kwargs.get("comment", "")
    f.write(f"{nat}\n")
    f.write(f"{comment}\n")

    if charges is not None:
        charges = np.asarray(charges).reshape(-1)
        if charges.shape[0] != nat:
            raise ValueError(f"charges must have length {nat}, got shape {charges.shape}")
    
    for i in range(nat):
        if charges is not None:
            #print(f"DEBUG: charges: {charges}")
            f.write(
                f"{symbols[i]:3} {coordinates[i,0]: 15.5e} {coordinates[i,1]: 15.5e} {coordinates[i,2]: 15.5e}    {float(charges[i]): .8f}\n"
            )
        else:
            f.write(
                f"{symbols[i]:3} {coordinates[i,0]: 15.5e} {coordinates[i,1]: 15.5e} {coordinates[i,2]: 15.5e}\n"
            )
    f.flush()
