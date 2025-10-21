from fennol.utils.atomic_units import UnitSystem
from fennol.utils.periodic_table import PERIODIC_TABLE
import numpy as np
import jax.numpy as jnp

__all__ = ['us','format_batch_conformations','update_conformation','get_composition_string']

us = UnitSystem(L="angstrom",T="ps",E="kcalpermol")


def format_batch_conformations(species,coordinates):
    """Format batch conformations into a dictionary for FeNNol model processing.

    Args:
     - species: array of shape (Nat,) with atomic species indices
     - coordinates: array of shape (M,Nat,3) with atomic coordinates for M conformations
    Returns:
     - conformation: dict with keys 'species', 'coordinates','natoms','batch_index'
    """
    
    M, Nat = coordinates.shape[0:2]
    species_batch = np.tile(species, M)  # (M * Nat,)
    coordinates_batch = coordinates.reshape(-1, 3)  # (M * Nat, 3)
    batch_index = np.repeat(np.arange(M,dtype=np.int32), Nat)  # (M * Nat,)
    natoms_batch = np.array([Nat] * M,dtype=np.int32)  # list of length M
    return {
        'species': jnp.array(species_batch,dtype=jnp.int32),
        'coordinates': jnp.array(coordinates_batch,dtype=jnp.float32),
        'natoms': jnp.array(natoms_batch,dtype=jnp.int32),
        'batch_index': jnp.array(batch_index,dtype=jnp.int32)
    }

def update_conformation(conformation, new_coordinates):
    """Update coordinates in a batch conformation dictionary.

    Args:
     - conformation: dict with keys 'species', 'coordinates','natoms','batch_index' 
        - new_coordinates: array of shape (M,Nat,3) with new atomic coordinates for M conformations 
    Returns:
     - updated_conformation: dict with updated 'coordinates'
    """
    return {**conformation, 'coordinates': jnp.asarray(new_coordinates.reshape(-1,3),dtype=jnp.float32)}

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