import numpy as onp
from .utils import us
import jax.numpy as jnp

def setup_repulsion_potential(target_species, projectile_species, use_jax=False):
    """Setup a ZBL repulsion potential between projectiles and target atoms.

    Args:
        - target_species: array of shape (N,) with target atomic species indices
        - projectile_species: atomic species index of the projectiles
    Returns:
        - repulsion_energies_and_forces: function that computes repulsion energies and forces
    """

    d = 0.46850
    p = 0.23
    alphas = onp.array((3.19980, 0.94229, 0.40290, 0.20162))
    cs = onp.array((0.18175273, 0.5098655, 0.28021213, 0.0281697))[None,:]

    Z = target_species.astype(onp.float32)
    Zb = float(projectile_species)
    Zpij = (Z**p + Zb**p)/d

    alphaij = -alphas[None,:]*Zpij[:,None]

    Zij = Z*Zb *us.BOHR/us.HARTREE 
    np = jnp if use_jax else onp

    def repulsion_energies_and_forces(coordinates, projectile_coordinates):
        """Compute ZBL repulsion energies and forces.
        Args:
            - coordinates: array of shape (batch_size,N,3) with target atomic coordinates
            - projectile_coordinates: array of shape (batch_size,3) with projectile coordinates
        Returns:
            - energies: array of shape (batch_size,) with repulsion energies
            - forces: array of shape (batch_size,N,3) with repulsion forces on target atoms
            - projectile_forces: array of shape (batch_size,3) with repulsion forces on projectiles
        """
        vecs = coordinates - projectile_coordinates[:,None,:]  # (batch_size,N,3)
        rij = np.linalg.norm(vecs,axis=-1)  # (batch_size,N)
        dirs = vecs/rij[:,:,None]  # (batch_size,N,3)
        phis = cs*np.exp(alphaij*rij[:,:,None])  # (batch_size,N,4)
        dphis_dr = alphaij*phis  # (batch_size,N,4)

        phi = np.sum(phis,axis=-1)  # (batch_size,N)
        dphi_dr = np.sum(dphis_dr,axis=-1)  # (batch_size,N)

        energies = np.sum(Zij[None,:]*phi/rij,axis=-1)  # (batch_size,)

        forces_magnitudes = Zij[None,:]*(dphi_dr/rij - phi/(rij**2))  # (batch_size,N)
        forces = -forces_magnitudes[:,:,None]*dirs  # (batch_size,N,3)
        projectile_forces = -np.sum(forces,axis=1)  # (batch_size,3)

        return energies, forces, projectile_forces

    return repulsion_energies_and_forces



