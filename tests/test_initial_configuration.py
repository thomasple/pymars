import numpy as np

from pymars.initial_configuration import read_initial_configuration, sample_velocities, remove_com_velocity, sample_projectiles
from fennol.utils.periodic_table import ATOMIC_MASSES
from pymars.utils import us


def test_read_initial_configuration(test_data_dir):
    filename = test_data_dir / "aspirin.xyz"
    species, coordinates = read_initial_configuration(filename)
    assert species.shape == (21,)
    assert coordinates.shape == (21,3)
    # symbols = "CCCCCCCOOOCCOHHHHHHHH"
    species_expected = np.array([6]*7+[8]*3+[6]*2+[8]*1+[1]*8,dtype=np.int32)
    assert np.all(species == species_expected)
    masses = np.array([12.]*7+[16.]*3+[12.]*2+[16.]*1+[1.]*8)
    totmass = masses.sum()
    center_of_mass = np.sum(coordinates * masses[:, np.newaxis], axis=0) / totmass
    assert np.allclose(center_of_mass, 0.0, atol=1e-3)

def test_boltzmann_velocity_distribution():
    temperature = 250.0  # Kelvin
    species_set = np.array([1, 6, 8], dtype=np.int32)
    masses_set = np.array([1., 12.0, 16.])/us.DA  # atomic masses
    nat=1000
    indices = np.random.choice(len(species_set), size=nat)
    species = species_set[indices]
    masses = masses_set[indices]
    velocities = sample_velocities(species, temperature)
    assert velocities.shape == (nat,3)
    Test = (masses[:, None] * velocities**2).mean()/us.K_B # in Kelvin


    Tstd_th = temperature / (1.5*nat)**0.5 # theoretical standard deviation of temperature fluctuations
    assert np.abs(Test - temperature) < 3*Tstd_th, "Temperature deviation too large"

def test_remove_com_velocity(test_data_dir):
    filename = test_data_dir / "aspirin.xyz"
    species, coordinates = read_initial_configuration(filename)
    masses = ATOMIC_MASSES[species]
    temperature = 300.0
    velocities = sample_velocities(species, temperature)

    velocities_corrected = remove_com_velocity(coordinates, velocities, species)

    # Check that COM velocity is removed
    totmass = masses.sum()
    com_velocity_after = np.sum(velocities_corrected * masses[:, None], axis=0) / totmass
    assert np.allclose(com_velocity_after, 0.0, atol=1e-4), f"COM linear velocity not removed, got {com_velocity_after.tolist()}"

    # Check that angular momentum is removed
    L_after = np.sum(np.cross(coordinates, velocities_corrected) * masses[:, None], axis=0)
    assert np.allclose(L_after, 0.0, atol=1e-4), "COM angular velocity not removed"

def test_projectile_initialization():
    n_projectiles = 1000
    projectile_species = 6  # Carbon
    temperature = 300.0  # Kelvin
    distance = 5.0  # angstrom
    max_impact_parameter = 1.0  # angstrom


    E = 1.5 * us.K_B * temperature  # average kinetic energy per particle
    excepted_speed = np.array([np.sqrt(2 * E / (ATOMIC_MASSES[projectile_species] / us.DA)),0.,0.])
    
    projectile_positions, projectile_velocities = sample_projectiles(
        n_projectiles, temperature, distance=distance, max_impact_parameter=max_impact_parameter,projectile_species=projectile_species
    )
    assert projectile_positions.shape == (n_projectiles, 3), "Incorrect projectile positions shape"
    assert projectile_velocities.shape == (n_projectiles, 3) , "Incorrect projectile velocities shape"
    assert np.allclose(projectile_velocities,excepted_speed[None,:]), f"projectile speeds do not match expected average speed"

    # Check that positions are within the specified impact parameter
    impact_parameters = np.linalg.norm(projectile_positions[:, 1:3], axis=1)
    assert np.all(impact_parameters <= max_impact_parameter + 1e-6), "Some projectiles exceed the maximum impact parameter"
    # Check that all projectiles are initialized at the correct distance
    assert np.allclose(projectile_positions[:, 0], -distance), "projectiles not initialized at the correct position"




