import numpy as np

from pymars.rotation_utils import uniform_orientation,apply_random_rotation,random_orientations

def test_uniform_orientation():
    n_points = 1000
    orientations = uniform_orientation(n_points)
    assert orientations.shape == (n_points, 3)
    norms = np.linalg.norm(orientations, axis=1)
    assert np.allclose(norms, 1.0), "All orientations should be unit vectors"

    dot_products = (orientations[:,None,:]*orientations[None,:,:]).sum(axis=-1)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))*180/np.pi
    angles = angles + 100*np.eye(n_points)
    min_angles = angles.min(axis=1)
    min_angle_mean = min_angles.mean()
    assert min_angle_mean <7, "Mean minimum angle should be less than 7 degrees"
    min_angle_std = min_angles.std()
    assert min_angle_std <0.1, "Std of minimum angle should be less than 2 degrees"

def test_selected_orientations():
    n_points = 1000
    orientations = uniform_orientation(n_points)

    ii = np.array([0,5,10,45,85])
    ori = uniform_orientation(n_points, indices=ii)
    assert ori.shape == (len(ii),3)
    ori2 = orientations[ii]
    assert np.allclose(ori, ori2), "Selected orientations should match"

    ii = 42
    ori = uniform_orientation(n_points, indices=ii)
    assert ori.shape == (3,)
    ori2 = orientations[ii]
    assert np.allclose(ori, ori2), "Selected orientation should match"


def test_random_orientations():
    n_samples = 100
    orientations = random_orientations(size=n_samples, n=1000)
    assert orientations.shape == (n_samples, 3)
    norms = np.linalg.norm(orientations, axis=1)
    assert np.allclose(norms, 1.0), "All orientations should be unit vectors"

    grid = uniform_orientation(1000)
    dot_products = (orientations[:,None,:]*grid[None,:,:]).sum(axis=-1).max(axis=1)
    assert np.all(dot_products > 0.99), "All orientations should be close to some grid point"


def test_apply_random_rotations():
    nvec = 6

    coords = np.random.rand(nvec,3) 
    dot_products = (coords[:,None,:] * coords[None,:,:]).sum(axis=-1)

    rotated_coords = apply_random_rotation(coords)
    assert rotated_coords.shape == (nvec,3)
    rotated_dot_products = (rotated_coords[:,None,:] * rotated_coords[None,:,:]).sum(axis=-1)
    assert np.allclose(dot_products, rotated_dot_products), "Dot products should be preserved under rotation"

    n_rotations = 5
    rotated_coords = apply_random_rotation(coords, n_rotations=n_rotations)
    assert rotated_coords.shape == (n_rotations, nvec, 3)
    for i in range(n_rotations):
        rotated_dot_products = (rotated_coords[i,:,None,:] * rotated_coords[i,None,:,:]).sum(axis=-1)
        assert np.allclose(dot_products, rotated_dot_products), "Dot products should be preserved under rotation"

