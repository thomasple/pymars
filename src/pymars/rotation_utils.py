import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import numpy as np

__all__ = ['apply_random_rotation', 'uniform_orientation', 'random_orientations']

def apply_random_rotation(coords, n_rotations=None):
    """ randomly rotate vectors.

    Args:
    - coords: array of shape (N,3)
    - n_rotations: number of random rotations to apply. If None, a single rotation is applied.
    Returns:
    - rotated_coordinates: rotated array of shape (n_rotations,N,3) or (N,3) if n_rotations is None
    """
    nrot = 1 if n_rotations is None else n_rotations
    angles = np.random.uniform(0,2*np.pi,size=(nrot,3))
    M = R.from_euler('zyx', angles, degrees=False).as_matrix() # (nrot,3,3)

    rotated_coordinates = np.einsum('rij,nj->rni', M, coords)
    return rotated_coordinates[0] if n_rotations is None else rotated_coordinates

def uniform_orientation(n,indices=None, offset=0.5):
    """Generate a spherical grid of n points using Fibonacci lattice."""
    one_vec = False
    if indices is None:
        i = np.arange(n)
    elif isinstance(indices, int):
        i = np.array([indices])
        one_vec = True
    else:
        i = np.asarray(indices)
    assert np.all((i >= 0) & (i < n)), "Indices must be in the range [0, n-1]"

    theta = (math.pi * i * (1 + math.sqrt(5))) % (2 * math.pi)
    phi = np.arccos(1 - 2 * (i + offset) / (n - 1 + 2 * offset))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    oris = np.stack((x, y, z), axis=-1)
    if one_vec:
        return oris[0]
    return oris

def generate_fibonacci_sphere(n=1000, offset=0.5):
    i = np.arange(n)
    theta = (math.pi * i * (1 + math.sqrt(5))) % (2 * math.pi)
    phi = np.arccos(1 - 2 * (i + offset) / (n - 1 + 2 * offset))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack((x, y, z), axis=-1)

def apply_fibonacci_random_rotation(atoms):
    coords = np.array([atom[1:] for atom in atoms])
    fib_points = generate_fibonacci_sphere(n=1000)
    axis = fib_points[np.random.randint(len(fib_points))]
    angle = np.random.uniform(0, 2*np.pi)
    r = R.from_rotvec(angle * axis)
    rotated_coords = r.apply(coords)
    rotated_atoms = [[atom[0]] + rotated_coords[i].tolist() for i, atom in enumerate(atoms)]
    return rotated_atoms

def random_orientations(size,n=10_000):
    """Generate random orientations by sampling from a uniform spherical grid."""
    indices = np.random.randint(0, n, size=size)
    orientations = uniform_orientation(n,indices=indices)
    return orientations
