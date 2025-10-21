# pymars

pymars is a small toolkit to prepare and run classical molecular collision simulations. It provides helpers to build initial configurations, set up short-range ZBL repulsion for projectile atoms and to run short collision simulations using FeNNol energy models.

The codebase is intentionally compact and focuses on collision setup and trajectory integration; FeNNol is used to provide model energies and forces while a simple ZBL-like repulsion model handles short-range interactions with incoming projectiles.

Key features
- Read and center XYZ geometries and produce batched conformations.
- Sample Maxwell–Boltzmann velocities and remove center-of-mass translation/rotation.
- Generate randomized orientations and uniformly sampled impact parameters for collision projectiles.
- Integrate dynamics with a Velocity-Verlet step combining FeNNol energies and a repulsive potential.

Requirements & installation
- Python >= 3.10
- Core runtime dependencies: `fennol` (see `pyproject.toml`), `numpy`, `scipy`, and `jax` (used by FeNNol workflows).

Install from source (editable install):

```bash
uv sync
source .venv/bin/activate
```


Quickstart (CLI)

Prepare a YAML configuration (example `config.yaml`):

```yaml
# config.yaml
initial_geometry: tests/aspirin.xyz   # path to an XYZ file or a dict with species/coordinates
model_file: /path/to/fennix_model     # file understood by fennol.FENNIX.load
simulation_time[ps]: 10.0              # total simulation time (ps)
dt[fs]: 1.                             # timestep  1 fs
batch_size: 1
temperature: 300.0
random_rotation: true
projectile_distance: 30.0                 # angstrom
max_impact_parameter: 1.0             # angstrom
output_file: trajectory.xyz
print_step: 100
```

Run the simulation using the provided CLI entry point:

```bash
pymars config.yaml
```

This will run a collision simulation and write a trajectory file (default `trajectory.xyz`).

Python API examples

Read an initial configuration and inspect shapes (example taken from the tests):

```python
from pymars.initial_configuration import read_initial_configuration

species, coordinates = read_initial_configuration('tests/aspirin.xyz')
print(species.shape, coordinates.shape)  # (21,), (21, 3)
```

Sample velocities and remove global translation/rotation:

```python
from pymars.initial_configuration import sample_velocities, remove_com_velocity

velocities = sample_velocities(species, temperature=300.0)
velocities = remove_com_velocity(coordinates, velocities, species)
```

Generate projectile initial positions and velocities:

```python
from pymars.initial_configuration import sample_projectiles

projectile_positions, projectile_velocities = sample_projectiles(
    n_projectiles=100,
    temperature=300.0,
    distance=20.0,
    max_impact_parameter=0.8,
    projectile_species=6,  # C
)
```

Create and run a short batched collision simulation (calls into FeNNol for model energies):

```python
from pymars.md import initialize_collision_simulation

config = {
    'initial_geometry': 'tests/aspirin.xyz',
    'model_file': '/path/to/fennix_model',
    'simulation_time': 1.0,
    'dt': 0.001,
    'batch_size': 2,
    'temperature': 300.0,
}

system = initialize_collision_simulation(config)
integrate = system['integrate']
coords = system['coordinates']
vels = system['velocities']
accs = system['accelerations']

# perform a few integration steps
for i in range(10):
    coords, vels, accs, energies = integrate(coords, vels, accs)
    print('Step', i, 'E_total mean:', energies.mean())
```

API summary
- `pymars.initial_configuration`
  - `read_initial_configuration(path)` — read and center an XYZ; returns `(species, coordinates)`.
  - `sample_velocities(species, temperature)` — Maxwell–Boltzmann velocities.
  - `remove_com_velocity(coordinates, velocities, species)` — remove COM translation & rotation.
  - `sample_projectiles(...)` — sample projectile positions and velocities.

- `pymars.rotation_utils`
  - `apply_random_rotation(coords, n_rotations=None)` — apply one or many random rotations.
  - `uniform_orientation(n, indices=None)` — generate a Fibonacci lattice on the unit sphere.
  - `random_orientations(size, n=10000)` — sample orientations from a grid.

- `pymars.repulsion`
  - `setup_repulsion_potential(target_species, projectile_species)` — returns a function computing short-range repulsive energies and forces (ZBL-like).

- `pymars.md`
  - `initialize_collision_simulation(simulation_parameters)` — prepare all arrays and return a dict with keys `integrate`, `coordinates`, `velocities`, `accelerations`, `species`, `masses`, `dt`, etc.

- `pymars.utils`
  - `us` — unit system (L=angstrom, T=ps, E=kcal/mol)
  - helpers to format batched conformations used by FeNNol models.

Notes on units
- Lengths are in angstroms, time in picoseconds (ps), energies in kcal/mol — see `pymars.utils.us` for the unit system used across the code.
- default units in the configuration YAML are expected to be in the unit system (A, ps, kcal/mol); e.g., time step `dt: 0.001` corresponds to 1 fs. Input units can be specified explicitly in the YAML keys (e.g., `dt[fs]: 1.0`).

Testing

Run the unit tests with pytest:

```bash
pytest -v
```

The `tests/` directory contains simple, focused tests that also serve as usage examples (for example, `tests/aspirin.xyz` is a small sample geometry used in the tests).

Development & contributing
- Please open issues or pull requests for new features or bug fixes.
- Add tests for new functionality and follow existing test patterns.

Acknowledgements
- This project uses FeNNol for model energies & forces and borrows unit conventions and utilities from that ecosystem.

License
- No license file is included in this repository. Add a LICENSE if you intend to publish the project.