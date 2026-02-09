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
# config.yaml - Structured format similar to NML input files

calculation_parameters:
  model: /path/to/fennix_model  # file understood by fennol.FENNIX.load

general_parameters:
  temperature: 300.0            # Temperature in Kelvin
  batch_size: 1                 # Number of parallel simulations
  seed: 12345                   # Random seed for reproducibility
  save_steps: 100               # Save trajectory frames every N steps
  save_energy: 100              # Save energy data every N steps

input_parameters:
  initial_geometry: tests/aspirin.xyz  # path to an XYZ file
  total_charge: 0               # Total molecular charge
  random_rotation: true         # Apply random rotation

projectile_parameters:
  projectile_flag: true         # Enable projectile collision simulation
  projectile_species: 18        # Atomic number (18 = Argon)
  projectile_temperature: 3000.0  # Projectile temperature (K)
  projectile_distance: 30.0     # Initial distance (angstrom)
  max_impact_parameter: 1.0     # Maximum impact parameter (angstrom)

thermostat_parameters:
  # Note: NVE_thermostat and LGV_thermostat cannot both be true
  NVE_thermostat: true          # Use NVE (microcanonical) ensemble
  LGV_thermostat: false         # Use Langevin thermostat
  gamma: 0.0                    # Friction constant (THz)

dynamic_parameters:
  dt_dyn: 1.0                   # Timestep in femtoseconds (fs)
  step_dyn: 10000               # Total number of MD steps

output_details:
  trajectory_file: trajectory.xyz  # Output trajectory file
  energies_file: energies.out   # Output energy file
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
    'calculation_parameters': {
        'model': '/path/to/fennix_model',
    },
    'input_parameters': {
        'initial_geometry': 'tests/aspirin.xyz',
        'total_charge': 0,
        'random_rotation': True,
    },
    'general_parameters': {
        'temperature': 300.0,
        'batch_size': 2,
        'seed': 12345,
        'save_steps': 100,
        'save_energy': 100,
    },
    'projectile_parameters': {
        'projectile_flag': True,
        'projectile_species': 18,
        'projectile_temperature': 3000.0,
        'projectile_distance': 30.0,
        'max_impact_parameter': 1.0,
    },
    'thermostat_parameters': {
        'NVE_thermostat': True,
        'LGV_thermostat': False,
        'gamma': 0.0,
    },
    'dynamic_parameters': {
        'dt_dyn': 1.0,
        'step_dyn': 1000,
    },
}

system = initialize_collision_simulation(config)
integrate = system['integrate']
coords = system['coordinates']
vels = system['velocities']
accs = system['accelerations']

# Setup energy output (optional)
energy_files = ['traj_0.out', 'traj_1.out']
save_energy = 100  # Write every 100 steps

# perform a few integration steps
for i in range(10):
    coords, vels, accs, energies = integrate(
        coords, vels, accs,
        step=i,
        energy_output_file=energy_files,
        energy_steps=save_energy
    )
    print('Step', i, 'E_total mean:', energies.mean())
```

**Energy Output Format:**

When `energy_output_file` is specified, pymars writes energy data in FeNNol-compatible format:

```
    Step   Time[fs]       Etot       Epot       Ekin    Temp[K]
       0      0.0000   -123.456789  -125.678901    2.222112    300.00
     100    100.0000   -123.457890  -124.567890    1.109999    250.12
```

Columns:
- `Step`: Integration step number
- `Time[fs]`: Time in femtoseconds
- `Etot`: Total energy (kcal/mol)
- `Epot`: Potential energy (kcal/mol)  
- `Ekin`: Kinetic energy (kcal/mol)
- `Temp[K]`: Temperature (Kelvin)

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
- This project is licensed under the terms of the GNU LGPLv3 license. See [LICENSE](https://github.com/thomasple/pymars/blob/main/LICENSE) for additional details.