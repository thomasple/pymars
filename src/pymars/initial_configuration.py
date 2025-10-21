import numpy as np

from fennol.utils.io import last_xyz_frame
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from .utils import us

__all__ = ['read_initial_configuration', 'sample_velocities', 'remove_com_velocity', 'sample_projectiles']


def read_initial_configuration(filename):
    """Read initial configuration from an XYZ file.

    Args:
     - filename: path to the XYZ file
    Returns:
     - symbols: list of atomic symbols
     - coordinates: array of shape (N,3) with atomic coordinates
    """
    symbols, coordinates, _ = last_xyz_frame(filename)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)
    masses = ATOMIC_MASSES[species]
    totmass = masses.sum()
    center_of_mass = np.sum(coordinates * masses[:, np.newaxis], axis=0) / totmass
    coordinates -= center_of_mass  # Center at center of mass
    return species, coordinates


def sample_velocities(species, temperature):
    """Sample velocities from Maxwell-Boltzmann distribution at given temperature.

    Args:
     - species: array of shape (N,) with atomic species indices
     - temperature: temperature in Kelvin
    Returns:
     - velocities: array of shape (N,3) with sampled velocities
    """

    kT = us.K_B * temperature
    masses = ATOMIC_MASSES[species] / us.DA
    stddev = np.sqrt(kT / masses)  # (N,)
    velocities = (
        np.random.normal(0.0, 1.0, size=(species.shape[0], 3)) * stddev[:, None]
    )

    return velocities


def remove_com_velocity(coordinates, velocities, species):
    """Remove center of mass linear and angular velocies

    Args:
     - coordinates: array of shape (N,3) with atomic coordinates (assumed relative to COM)
     - velocities: array of shape (N,3) with atomic velocities
     - species: array of shape (N,) with atomic species indices
    Returns:
     - velocities: array of shape (N,3) with adjusted velocities (rotational and translational components removed)
    """

    masses = ATOMIC_MASSES[species]
    totmass = masses.sum()

    com_velocity = np.sum(velocities * masses[:, None], axis=0) / totmass
    velocities = velocities - com_velocity  # Remove center of mass velocity

    # Total angular momentum L = sum_i r_i x (m_i * v_i)
    # shape: (3,)
    L = np.sum(np.cross(coordinates, velocities) * masses[:, None], axis=0)

    # Build the 3x3 moment of inertia tensor I = sum_i m_i [ (r_i·r_i) I_3 - r_i r_i^T ]
    rr = coordinates[:, :, None] * coordinates[:, None, :] 
    r2 = np.trace(rr, axis1=1, axis2=2)[:, None, None] * np.eye(3)[None, :, :]
    I = np.sum(masses[:, None, None] * (r2 - rr), axis=0)

    # Invert inertia tensor to get I^{-1}. This assumes I is non-singular (system not collinear / degenerate).
    I_inv = np.linalg.inv(I)

    # Angular velocity vector omega that corresponds to angular momentum L: omega = I^{-1} L
    omega = I_inv @ L

    # Subtract the rigid-body rotational velocity at each position: v_rot = omega x r
    # After this correction the net angular momentum should be (numerically) zero.
    vcorr = np.cross(omega, coordinates)
    velocities -= vcorr

    return velocities


def sample_projectiles(
    n_projectiles, temperature, distance, max_impact_parameter=0.5, projectile_species=18
):
    """Sample projectile initial positions and velocities for collision simulations.

    Args:
     - n_projectiles: number of projectiles to sample
     - temperature: temperature in Kelvin for velocity sampling
    - distance: initial distance of projectiles from origin along -x direction (in angstroms)
        - max_impact_parameter: maximum impact parameter in angstroms (default 0.5)
    Returns:
     - projectile_positions: array of shape (n_projectiles,3) with initial positions
     - projectile_velocities: array of shape (n_projectiles,3) with initial velocities
    """

    E = 1.5 * us.K_B * temperature  # average kinetic energy per particle
    mass_projectile = ATOMIC_MASSES[projectile_species] / us.DA  # in amu
    v_magnitude = np.sqrt(2 * E / mass_projectile)
    projectile_velocities = np.zeros((n_projectiles, 3), dtype=np.float32)
    projectile_velocities[:, 0] = v_magnitude  # projectiles move along x

    # Sample impact parameters uniformly within a circle of radius max_impact_parameter
    r = max_impact_parameter * np.sqrt(np.random.uniform(0, 1, size=n_projectiles))
    theta = np.random.uniform(0, 2 * np.pi, size=n_projectiles)
    y = r * np.cos(theta)
    z = r * np.sin(theta)
    projectile_positions = np.zeros((n_projectiles, 3), dtype=np.float32)
    projectile_positions[:, 0] = -distance  # start at -distance along x
    projectile_positions[:, 1] = y
    projectile_positions[:, 2] = z

    return projectile_positions, projectile_velocities



    
