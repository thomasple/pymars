# pymars

pymars is a compact toolkit to prepare and run reactive trajectories simulations. 
It provides helpers to build initial configurations, set up short-range ZBL repulsion for projectile atoms and run short collision simulations using FeNNol energy models.

The project focuses on collision setup and trajectory integration. 
FeNNol provides model energies and forces while a simple ZBL-like repulsion model handles short-range interactions with incoming projectiles.

Key features
- Read and center XYZ geometries and produce batched conformations.
- Sample Maxwell–Boltzmann velocities and remove center-of-mass translation/rotation.
- Generate randomized orientations and uniformly sampled impact parameters for collision projectiles.
- Integrate dynamics with a Velocity-Verlet step combining FeNNol energies and a repulsive potential.
- Batch-mode support (run multiple trajectories in parallel) and reproducible runs via `seed`.

Requirements & installation
- Environment with Python >= 3.10
- Core runtime dependencies: `fennol` (see `pyproject.toml`), `numpy`, `scipy`, and `jax` (used by FeNNol workflows) and their dependencies. 

See `installation_guide.txt` for a step-by-step installation and troubleshooting notes.

Quickstart (CLI)

Prepare a YAML configuration (example `input.yaml`):

```yaml
# PyMARS Simulation Configuration Input File

calculation_parameters:
  device: cpu                         # Options: 'cpu', 'cuda:0', 'gpu:0', etc. Default: cpu
  model: "/path/to/fennix/model.fnx"  # Path to FENNIX model file
  double_precision: false             # Use double precision (float64) instead of single (float32). Default: false
  save_initial: true                  # If true, saves the initial configuration as <xyzfilename>.dyn.init.npz or batch version <xyzfilename>.batchdyn.init.npz
  restart_traj: false                 # If true, continue from <xyzfilename>.dyn.restart.npz if it exists; always saves last state

general_parameters:
  temperature: 300.0   # Temperature in Kelvin
  batch_size: 1        # Number of parallel simulations (trajectories)
  seed: 123456789      # Random seed for reproducibility (0 to 999,999,999). If not set, a random seed is used.
  save_steps: 100      # Save trajectory frames every N steps
  save_energy: 100     # Save energy data every N steps
  save_summary: 1000   # Write summary statistics every N steps (must be >= save_energy)

input_parameters:
  initial_geometry: "tests/aspirin.xyz"  # Path to initial XYZ geometry
  total_charge: 0                        # Total molecular charge
  random_rotation: true                  # Apply random rotation to initial configuration

projectile_parameters:
  projectile_flag: true           # Enable projectile collision simulation
  projectile_species: 18          # Atomic number (18 = Argon)
  projectile_temperature: 3000.0  # Projectile temperature in Kelvin
  projectile_distance: 20.0       # Initial projectile distance in angstrom
  max_impact_parameter: 0.5       # Maximum impact parameter in angstrom

thermostat_parameters:
  # Currently only NVE (microcanonical, energy-conserving) ensemble is supported
  NO_thermostat: true  # Use NVE (microcanonical) ensemble
  # Future thermostat options (not yet implemented):
  # LGV_thermostat: false  # Use Langevin thermostat
  # gamma: 0.0  # Friction constant in THz (for Langevin thermostat)

dynamic_parameters:
  dt_dyn: 1.0      # Timestep in femtoseconds (fs)
  step_dyn: 10000  # Total number of MD steps

output_details:
  trajectory_file: "trajectory.xyz"  # Output trajectory file
  #[...]
  
  energies_file: "energies.out"  # Output energy file (recommended to use .out extension)
  # [...]
  
  track_variance: false     # If true, add a Var(eV^2) column to energies.out using the model's etot_ensemble_var.
                            # Requires a FeNNol model that exposes 'etot_ensemble_var' in its auxiliary output.
                            # Automatically disabled when batch_size > 1.
  summary_file: "summary.out"   # Output summary statistics file (recommended to use .out extension) [...]
```

Run the simulation using the provided CLI entry point:

```bash
pymars input.yaml
```

This will run a collision simulation and write a trajectory file (default `trajectory.xyz`).

Recent changes (1.7.4)
-----------------------------------

The last update changed both the examples and the PhoBOOS runtime behavior:

- `phoboos` now prints total runtime on normal exit, instead of only a generic termination message.
- The vibrational frequency workflow now prints the frequency sequence one mode per line after the formatted report.
- PhoBOOS transition-state search now uses separate search-stage and refinement-stage parameters.
- The PhoBOOS example inputs were reorganized into dedicated folders for `singlepoint-freq`, `optimization`, `optts`, and `tssearch`.
- The main `pymars` example input now documents verbose output control, initial-state replay, and restart handling.

The configuration reference below matches the example files in `examples/`.

### Main `pymars` example (`examples/pymars/input.yaml`)

The main collision-simulation example currently includes these sections and keys:

- `calculation_parameters`
  - `device`: Compute device, such as `cpu` or `cuda:0`.
  - `model`: Path to the FENNIX model file.
  - `double_precision`: Use float64 coordinates and velocities when true.
  - `save_initial`: Save the initial state to a `.npz` file when true.
  - `start_from_init`: Load the saved initial state when present.
  - `restart_traj`: Resume from a restart file when present.

- `general_parameters`
  - `temperature`: Temperature in Kelvin.
  - `batch_size`: Number of trajectories.
  - `seed`: Random seed or per-trajectory seed list.
  - `save_steps`: Trajectory save frequency.
  - `save_energy`: Energy save frequency.
  - `save_summary`: Summary save frequency.

- `input_parameters`
  - `initial_geometry`: Starting XYZ geometry.
  - `total_charge`: Total molecular charge.
  - `random_rotation`: Apply a random rotation to the initial configuration.

- `projectile_parameters`
  - `projectile_flag`: Enable projectile collision setup.
  - `projectile_species`: Projectile atomic number.
  - `projectile_temperature`: Projectile temperature in Kelvin.
  - `projectile_distance`: Initial projectile distance in angstrom.
  - `max_impact_parameter`: Maximum impact parameter in angstrom.

- `thermostat_parameters`
  - `NO_thermostat`: Use NVE / microcanonical dynamics.
  - `LGV_thermostat`: Legacy Langevin toggle kept for compatibility.
  - `gamma`: Langevin friction constant.

- `dynamic_parameters`
  - `dt_dyn`: Timestep in femtoseconds.
  - `step_dyn`: Number of integration steps.

- `output_details`
  - `verbose`: Print extra run information.
  - `trajectory_file`: Trajectory filename.
  - `energies_file`: Energy output filename.
  - `track_variance`: Add a `Var(eV^2)` column when the model exposes `etot_ensemble_var`.
  - `summary_file`: Summary output filename.

PhoBOOS commands (`spt`, `freq`, `opt`, `optts`, `tssrc`)
--------------------------

`phoboos` is the geometry-focused companion CLI (entry point: `phoboos = pymars.phoboos:main`).
It reads an input YAML and runs either single-point energy evaluation or geometry optimization
using a FeNNIX model.

PhoBOOS input file structure
----------------------------

The CLI reads the following YAML sections. Only `calculation_parameters` and `input_parameters`
are required for every command; the other sections are used by specific subcommands.

### `calculation_parameters`

Settings that control the FeNNIX model and execution environment.

- `model`: Path to the FeNNIX model file (`.fnx`). Required.
- `device`: Compute device for JAX/FeNNol. Common values are `cpu`, `cuda:0`, `gpu:0`.
  If omitted, `cpu` is used.

### `input_parameters`

Geometry and system information shared by all PhoBOOS commands.

- `initial_geometry`: Path to the starting XYZ geometry. Required.
- `final_geometry`: Path to the target XYZ geometry for `tssrc`. Required only for `tssrc`.
- `total_charge`: Total molecular charge. Default: `0`.

### `optimization_parameters`

Used by `opt`, `optts`, and `tssrc`.

- `tolerance`: Convergence threshold for the geometry optimizer in `opt`. Default: `1e-2`.
- `f_max`: Force threshold used by `optts`. The example input sets this to `0.2`.
- `dx_max`: Maximum step size used by the geometry optimizer. Default: `0.2`.
- `dt_dyn`: Timestep in femtoseconds for `opt`. It is converted internally to picoseconds. Default: `2.0`.
- `max_steps`: Maximum number of optimizer steps. Default: `1000` for TS workflows and `10000` for `opt`.
- `save_steps`: Save frequency for optimization trajectory output. Default: `-1` (final structure only).

### `search_parameters`

Used by `tssrc`.

- `effort`: Transition-state search level. Supported values are `low`, `medium`, and `high`.
- `f_max`: Force threshold for the NEB search stage. The example input sets this to `0.3`.
- `max_steps`: Maximum number of NEB search steps. The example input sets this to `1000`.

### Command behavior overview

### Command: `phoboos spt`

Single-point calculation:

```bash
phoboos spt examples/phoboos_spt/input.yaml
```

What it does:
- Loads `input_parameters.initial_geometry` (XYZ)
- Loads the FeNNIX model from `calculation_parameters.model`
- Evaluates energy (and prints corresponding output/log info)
- Does not run an optimization trajectory

Input parameters used by `phoboos spt`:
- `calculation_parameters.model`: FeNNIX model path
- `calculation_parameters.device`: Optional device selection
- `input_parameters.initial_geometry`: Starting XYZ geometry
- `input_parameters.total_charge`: Optional total charge

### Command: `phoboos freq`

Single-point energy plus harmonic vibrational analysis:

```bash
phoboos freq examples/phoboos_spt/input.yaml
```

What it does:
- Reads the geometry and sets up a FENNIX ASE calculator.
- Centers the structure on its center of mass.
- Rotates into the principal-axes frame before the vibrational calculation.
- Calls ASE `Vibrations` to compute the finite-difference Hessian and normal modes.
- Prints a formatted vibrational report with frequencies, reduced masses, force constants, and per-mode energies.

Implementation details:
- `run_freq` converts the input geometry to an ASE `Atoms` object, attaches a `FENNIXCalculator(model, inputs=..., gpu_preprocessing=True)`, and then calls `Vibrations(atoms)`.
- The code explicitly normalizes the geometry before the Hessian step: center of mass removal and principal-axis alignment are done to reduce rotational/translational artifacts.
- The report is built in `_format_report()`, which reshapes each mode, estimates reduced masses, converts to force constants, and prints real frequencies with imaginary modes kept negative.
- The final output includes the block report and a one-line-per-mode frequency list in cm^-1, matching the runtime behavior currently implemented in `pymars.phoboos.frequencies`.

Input parameters used are the same as `phoboos spt`:
- `calculation_parameters.model`: FeNNIX model path
- `calculation_parameters.device`: Optional device selection
- `input_parameters.initial_geometry`: Starting XYZ geometry
- `input_parameters.total_charge`: Optional total charge

### Command: `phoboos opt`

Geometry optimization (iterative structure update):

```bash
phoboos opt examples/phoboos_opt/input.yaml
```

What it does:
- Loads the initial geometry and FeNNIX model.
- Runs an iterative geometry optimization using the FIRE algorithm.
- Writes an optimization trajectory when `save_steps > 0`.
- Stops at convergence (`tolerance`) or when `max_steps` is reached.

Implementation notes:
- The actual optimizer is the FENNOL helper `optimize_fire2`, not a custom minimizer implemented in pymars.
- The code passes `atol=tolerance`, `dt=dt_ps`, `Nmax=max_steps`, `logoutput=True`, `keep_every=keep_every`, and `max_disp=dxmax` to the FIRE routine.
- In practice, FIRE is a damped, velocity-assisted gradient descent: it maintains a pseudo-velocity, scales the step dynamically to accelerate downhill motion, and reduces or resets the velocity when the descent direction turns unfavorable.
- The `dxmax` parameter curbs the maximum displacement per step, which keeps large jumps from occurring during the line-search/relaxation phase.
- The final energy is re-evaluated with the FENNIX model and the optimized geometry is written to `<basename>.opt.xyz` unless `outfile` is explicitly set.

Input parameters used by `phoboos opt`:
- `calculation_parameters.model`: FeNNIX model path
- `calculation_parameters.device`: Optional device selection
- `input_parameters.initial_geometry`: Starting XYZ geometry
- `input_parameters.total_charge`: Optional total charge
- `optimization_parameters.tolerance`: Convergence threshold
- `optimization_parameters.dx_max`: Maximum step size
- `optimization_parameters.dt_dyn`: Timestep in femtoseconds
- `optimization_parameters.max_steps`: Maximum number of optimizer steps
- `optimization_parameters.save_steps`: Trajectory save frequency

Optimization parameters used by `phoboos opt`:
- `optimization_parameters.tolerance` (default: `1e-2`)
- `optimization_parameters.dx_max` (default: `0.2`)
- `optimization_parameters.dt_dyn` in fs (default: `2.0`, internally converted to ps)
- `optimization_parameters.max_steps` (default: `10000`)
- `optimization_parameters.save_steps` (default: `-1`, save final only)

### Command: `phoboos optts`

Transition-state optimization (Sella / RS-PRFO):

```bash
phoboos optts examples/phoboos_spt/input.yaml
```

What it does:
- Starts from a guess geometry and optimizes it as a first-order saddle point.
- Uses ASE/Sella rather than a minimum-energy optimizer.
- Writes the optimized TS guess to a `.tsopt.xyz` file and then evaluates its energy and vibrational spectrum.

Implementation details:
- The solver is `Sella(atoms, order=1, trajectory=..., logfile=...)`.
- `order=1` tells Sella to optimize for a transition state (first-order saddle point), while `order=0` would target a minimum.
- The CLI passes `fmax` and `max_steps` from `optimization_parameters` to `opt.run(fmax=fmax, steps=max_steps)`.
- After the optimization, the code evaluates a single-point energy and runs `run_freq` on the final structure.

### Command: `phoboos tssrc`

Transition-state search using NEB, then TS refinement:

```bash
phoboos tssrc examples/phoboos_spt/input.yaml
```

What it does:
- Loads `initial_geometry` and `final_geometry`.
- Aligns the endpoints before creating the band.
- Builds an NEB path between the reactant and product states using ASE `NEB` images.
- Runs a loose NEB relaxation, identifies the highest-energy image, and then performs a climbing-image refinement.
- For `effort=high`, the path is refined around the barrier with a zoomed refinement stage.
- Transfers the resulting TS guess to a Sella optimization for final saddle-point convergence.

Implementation details:
- `run_tssrc` creates a list of image structures and attaches a `FENNIXCalculator` to each image.
- It uses `NEB(images, method='improvedtangent', climb=False)`, interpolates the path with `idpp`, and relaxes the band using `LBFGS`.
- The code then activates `neb.climb = True` and re-runs `LBFGS` to push the highest-energy image uphill while relaxing the rest of the band.
- The highest-energy image is selected from the resulting path and used as the TS guess.
- For `mode='tight'`, the code creates a local refinement band around the peak and repeats the NEB relaxation on a reduced segment.
- The final TS guess is written out as `ts_guess.xyz`, and the CLI then calls `run_tsopt` on that geometry for the final saddle-point optimization.

Input parameters used by `phoboos tssrc`:
- `calculation_parameters.model`: FeNNIX model path
- `calculation_parameters.device`: Optional device selection
- `input_parameters.initial_geometry`: Reactant XYZ geometry
- `input_parameters.final_geometry`: Product XYZ geometry
- `input_parameters.total_charge`: Optional total charge
- `search_parameters.effort`: One of `low`, `medium`, or `high`
- `search_parameters.f_max`: Force threshold for the NEB search stage
- `search_parameters.max_steps`: Maximum number of NEB search steps
- `optimization_parameters.f_max`: Force threshold for the final TS optimization stage
- `optimization_parameters.max_steps`: Maximum number of final TS optimization steps

Notes:
- `phoboos` writes internal log output to `<initial_geometry_basename>.out` next to the input YAML.
- Available subcommands include `spt`, `freq`, `opt`, `optts`, and `tssrc`.

## On the Nudged Elastic Band Transition state search

The transition-state workflow implemented here is a two-level ASE-based search:

1. A coarse NEB search over a path connecting reactant and product images.
2. A final `Sella` saddle-point refinement on the best TS guess.

The Nudged Elastic Band (NEB) method is used to find a minimum energy path (MEP) connecting two minima on the potential-energy surface. In practice, the code builds a chain of structural images between the endpoints, interpolates them with an IDPP guess, and relaxes the path while preserving even spacing along the path. The highest-energy image is then treated as the candidate transition state.

The implementation is not a fully custom NEB code; it uses ASE `NEB` and `LBFGS` directly. In `run_tssrc`, the path is initialized as:

```python
images = [atoms1] + [atoms1.copy() for _ in range(n_images)] + [atoms2]
neb = NEB(images, method='improvedtangent', climb=False)
neb.interpolate('idpp')
opt = LBFGS(neb, trajectory=str(binneb_file), logfile=log_file)
opt.run(fmax=fmax_neb, steps=max_steps)
```

This is followed by a climbing-image stage:

```python
neb.climb = True
opt = LBFGS(neb, trajectory=str(binneb_file), logfile=str(log_file), maxstep=0.1)
opt.run(fmax=fmax_ci, steps=max_steps)
```

This phase is the key CI-NEB step: the image with the largest energy is driven uphill along the local tangent while the other images relax downhill in the orthogonal directions. The code then selects the highest-energy image as the rough transition-state guess.

The `effort` setting controls the search strategy:
- `low`: 12 images, loose CI-NEB search.
- `medium`: 20 images, loose CI-NEB search.
- `high`: 20 images plus a zoom refinement around the peak image before handing the best candidate to the final TS optimizer.

After the NEB search, the code passes the saddle guess into Sella (`order=1`) for a final transition-state optimization. This is the same RS-PRFO-style optimization used by `phoboos optts` and is intended to converge the approximate NEB image into a true first-order saddle point.

Python API examples:

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
        'NO_thermostat': True,
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

**Per-trajectory energy filenames (batch mode):**

When running with `batch_size > 1` and a single configured filename, pymars inserts the trajectory index before the file extension using `os.path.splitext`. Examples:
- `energies.out` → `energies_0.out`, `energies_1.out`, …
- `energies` → `energies_0`, `energies_1`, …
- `my.out.file` → `my.out_0.file`, `my.out_1.file`, …

Batch output and SIM export
---------------------------

When running batch simulations (`batch_size > 1`), pymars automatically organizes per-trajectory outputs into timestamped simulation folders (`SIM00000`, `SIM00001`, …):

- **SIM folder creation**: One folder per trajectory, numbered sequentially based on existing folders (e.g., if `SIM00000` and `SIM00001` exist, the next batch starts at `SIM00002`).
- **Artifact organization**: Per-trajectory output files are moved and renamed into their respective SIM folder:
  - Trajectory file (e.g., `trajectory_0.xyz` → `trajectory.xyz` in `SIM00000`)
  - Energy file (e.g., `energies_0.out` → `energies.out` in `SIM00000`)
  - Summary file (e.g., `summary_0.out` → `summary.out` in `SIM00000`)
  - This makes each SIM folder self-contained with consistently-named output files.
- **Input YAML copy with per-trajectory seed**: The input configuration YAML is copied into each SIM folder and its `seed:` line is rewritten to contain that trajectory's assigned seed. This enables easy reproduction of any single trajectory from the batch.

Example batch output structure after a 3-trajectory run:
```
SIM00000/
  trajectory.xyz
  energies.out
  summary.out
  input.yaml          # seed: 11
SIM00001/
  trajectory.xyz
  energies.out
  summary.out
  input.yaml          # seed: 22
SIM00002/
  trajectory.xyz
  energies.out
  summary.out
  input.yaml          # seed: 33
```

Restart files and variance tracking
---------------------------------

Restart files
: When a run finishes pymars saves the last state to a restart file so you can resume later. The restart filename is derived from the input geometry basename and is written next to the input YAML (for example, when your YAML is `examples/input.yaml` the restart file will be `examples/<base>.dyn.restart.npz`). numpy's `savez` is used, so the on-disk file ends with `.npz`.

To resume a run set `calculation_parameters.restart_traj: true` in your YAML. pymars will look for `<base>.dyn.restart` and `<base>.dyn.restart.npz` next to the YAML and load positions/velocities/accelerations and the saved `step` if present.

Variance tracking (model-provided)
: If your FeNNol model exposes an ensemble variance per-frame named `etot_ensemble_var` pymars can record it in the energy output. Enable this with:

```yaml
output_details:
  energies_file: "energies.out"
  track_variance: true
```

When enabled, the energies file will include an additional column `Var(eV^2)` (the model-provided variance; units are left as eV^2). The variance value is written only in the energy output file (`energies.out`) and is **not** printed in the regular console summary to avoid spurious `None` values when the model does not provide variance.

Notes:
- If your model does not provide `etot_ensemble_var`, the `Var(eV^2)` column will contain `None` for that frame.
- `track_variance` is meaningful only for single-trajectory runs (batch_size == 1). If `track_variance: true` and `batch_size > 1` pymars will disable the option and print a warning because the ensemble variance per-frame is not defined across parallel trajectories.


```

API summary

The signatures below match the current(1.7.5) implementation in the dev package and include the optional arguments exposed by the source docstrings.

- `pymars.initial_configuration`
  - `read_initial_configuration(filename)` — read and center an XYZ; returns `(species, coordinates)`.
  - `sample_velocities(species, temperature, rng=None, dtype=None)` — Maxwell–Boltzmann velocities with optional RNG/dtype control.
  - `remove_com_velocity(coordinates, velocities, species)` — remove COM translation and rigid-body rotation.
  - `sample_projectiles(n_projectiles, temperature, distance, max_impact_parameter=0.5, projectile_species=18, rng=None, dtype=None)` — sample projectile positions and velocities for collision setups.

- `pymars.rotation_utils`
  - `apply_random_rotation(coords, n_rotations=None, rng=None)` — apply one or many random rotations.
  - `uniform_orientation(n, indices=None, offset=0.5)` — generate a Fibonacci-lattice spherical sampling grid.
  - `apply_fibonacci_random_rotation(coords, n=1000, n_rotations=None)` — generate a rotation from Fibonacci-sampled axes.
  - `random_orientations(size, n=10000)` — sample random orientations from a uniform spherical grid.

- `pymars.repulsion`
  - `setup_repulsion_potential(target_species, projectile_species, use_jax=False)` — returns a function computing short-range repulsive energies and forces (ZBL-like).

- `pymars.md`
  - `initialize_collision_simulation(simulation_parameters, verbose=True)` — prepare all arrays and return a dict with keys including `integrate`, `coordinates`, `velocities`, `accelerations`, `species`, `masses`, `dt`, and batch metadata. The function supports both nested and legacy flat configuration layouts.

- `pymars.utils`
  - `us` — unit system (L=angstrom, T=ps, E=kcal/mol)
  - `format_batch_conformations(species, coordinates, total_charge=0)` — pack batched coordinates into the FeNNol conformation dict format.
  - `update_conformation(conformation, new_coordinates)` — replace coordinates in a packed conformation.
  - `get_composition_string(species)` — build a composition string such as `O1_H2`.
  - `write_xyz_frame(...)` — write a single XYZ frame, including optional charges and a custom comment line.

Initial state saving and reproducibility
------------------------------------------

Initial conditions replay (batch → single trajectory)
: When running batch simulations (`batch_size > 1`), pymars saves the exact initial conditions (positions, velocities, accelerations) for each trajectory:
- **Single trajectory**: saved to `<xyzfilename>.dyn.init.npz` in the simulation folder
- **Batch**: saved to `<xyzfilename>.batchdyn.init.npz` in the parent directory

These files are controlled by the `save_initial` flag (default `true`). When set to `false`, no initial state files are written.

To reproduce a single trajectory from a batch run:
1. Locate the trajectory's `<xyzfilename>.dyn.init.npz` file in its `SIMXXXXX` folder
2. Run with `batch_size: 1` and the per-trajectory seed; pymars will automatically load the saved initial state (when `restart_traj: false`)
3. This ensures bit-identical initial conditions

**Important note**: Floating-point arithmetic depends on batch shape (batch_size). Running a trajectory extracted from a batch with `batch_size: 1` will have slightly different numerical paths due to:
- Different vectorization order in JAX/NumPy operations
- Different floating-point reduction sequences

For bit-wise reproducibility across trajectories and batches, use `save_initial: true` (default).

Seeding and random number generation
: pymars uses NumPy's modern Generator API (`np.random.default_rng(seed)`) with per-trajectory seeding:

**Seed configuration**:
- User provides one seed per trajectory via `general_parameters.seed`:
  - Single scalar (for `batch_size: 1`) → used directly
  - List/tuple/array → one seed per trajectory (must match batch_size)
  - Comma-separated string (for batch runs) → split into per-trajectory seeds
  - Missing/`null` → random unique seeds are generated per trajectory
- For batch runs (`batch_size > 1`), all seeds must be **unique**; duplicates in user-provided lists trigger an error; duplicates in auto-generated lists are regenerated
- Output logs print the resolved per-trajectory seed list for reference

**How seeds are used**:
Each trajectory gets its own NumPy Generator seeded with `seed_list[b]`. This RNG drives:
- Random orientation sampling (via `scipy.spatial.transform.Rotation.random(..., random_state=rng_b)`)
- Initial velocity sampling (Maxwell–Boltzmann via `rng_b.normal(...)`)
- Projectile initial conditions sampling
- Any other stochastic initialization

Same seed → identical internal RNG state → **identical random sequences** (assuming same NumPy/SciPy versions and execution order).

**NumPy Generator details**:
Modern NumPy (`>= 1.17`) uses `Generator(PCG64)` by default:
- `np.random.default_rng(seed)` creates a Generator with a PCG64 BitGenerator
- The seed is used to initialize the BitGenerator's internal state
- Each call to the RNG (e.g., `.normal()`, `.uniform()`) applies deterministic transformations to the internal state
- Same seed → same state sequence → identical outputs

**Important**: The old global seed API (`np.random.seed(...)`) is not used; all randomness is seeded per-trajectory via independent Generator instances.

**Reproducibility metadata**:
At startup, pymars prints:
- NumPy version and SciPy version
- Hardware/platform info (OS, release, machine, processor)

Recording these alongside your seed list ensures **full reproducibility** across machines and time:

```
# Per-trajectory seeds:
# trajectory         seed
#          0    1234567890
# NumPy version: 1.26.4
# SciPy version: 1.11.4
# Hardware: Linux 5.15.0-1234 x86_64 | processor=Intel(R) Core(TM) i7-11700K AT
# Coordinate dtype: float32
```

**Note on reproducibility across batch sizes**: Different batch sizes lead to different JAX/NumPy vectorization paths and floating-point reduction orders, even with identical seeds. For guaranteed numerical reproducibility, use identical batch sizes and versions.

Floating-point precision
: By default, pymars uses single precision (float32) for coordinates, velocities, and rotations. Set `calculation_parameters.double_precision: true` to use float64:

```yaml
calculation_parameters:
  double_precision: true  # Use float64 for coordinates and velocities
```

Masses and atomic species remain in float32 for efficiency. The flag is printed at runtime:
```
# Coordinate dtype: float64
```

**Note**: Double precision reduces but does not eliminate numerical divergence in chaotic dynamics (MD trajectories). For guaranteed reproducibility, use identical batch shapes and NumPy/SciPy versions.

Program output:
the program implements internal logging to a file called <input structure filename>.out

Notes on units
- Lengths are in angstroms, time in picoseconds (ps), energies in kcal/mol — see `pymars.utils.us` for the unit system used across the code.
- default units in the configuration YAML are expected to be in the unit system (A, ps, kcal/mol); e.g., time step `dt: 0.001` corresponds to 1 fs. Input units can be specified explicitly in the YAML keys (e.g., `dt[fs]: 1.0`).

Thermostat configuration
- `thermostat_parameters.NO_thermostat: true` (default) — use NVE (microcanonical, energy-conserving) ensemble
- The old name `NVE_thermostat` is still supported for backward compatibility but `NO_thermostat` is preferred

Testing

Run the unit tests with pytest:

```bash
pytest -v
```

The `tests/` directory contains simple, focused tests that also serve as usage examples:
- `tests/aspirin.xyz` — small sample geometry used in the tests.
- `tests/test_batch_sim_export.py` — regression test for batch SIM folder creation, artifact organization, per-trajectory seed rewriting, and initial state file handling (`<prefix>.dyn.init.npz`).

Development & contributing
- Please open issues or pull requests for new features or bug fixes.
- Add tests for new functionality and follow existing test patterns.

Acknowledgements
- This project uses FeNNol for model energies & forces and borrows unit conventions and utilities from that ecosystem.

License
- This project is licensed under the terms of the GNU LGPLv3 license. See [LICENSE](https://github.com/thomasple/pymars/blob/main/LICENSE) for additional details.