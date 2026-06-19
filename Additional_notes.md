# Technical Notes: NumPy RNG Seeding and JAX CPU Parallelization

## Table of Contents

1. [NumPy Random Number Generation and Seeding](#1-numpy-random-number-generation-and-seeding)
2. [Seed Parsing, Storage, and Consumption in the MD Workflow](#2-seed-parsing-storage-and-consumption-in-the-md-workflow)
3. [JAX CPU Parallelization](#3-jax-cpu-parallelization)

---

## 1. NumPy Random Number Generation and Seeding

### 1.1 Generator API

NumPy is not using the legacy global RNG (`np.random.seed(...)`). It uses the newer Generator API:

```python
rng = np.random.default_rng(seed)
```

This distinction matters because the underlying mechanics differ from the legacy API.

### 1.2 What `np.random.default_rng(seed)` Does

When called as:

```python
rng = np.random.default_rng(1234)
```

NumPy performs the following steps:

1. Takes the integer seed (`1234`).
2. Uses it to initialize a `BitGenerator`.
3. Wraps that `BitGenerator` in a high-level `Generator` object.

By default, modern NumPy uses a `Generator` backed by `PCG64`. Internally, this is approximately equivalent to:

```python
from numpy.random import Generator, PCG64

bitgen = PCG64(1234)
rng = Generator(bitgen)
```

### 1.3 Pseudo-Random Number Generators (PRNGs)

The numbers produced are not truly random. A PRNG:

- Starts from an internal state.
- Applies deterministic mathematical transformations.
- Produces a sequence that statistically resembles randomness.

Same seed → same internal state → same sequence:

```python
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)

rng1.normal()
rng2.normal()
```

These two calls always return identical outputs. This is the basis for reproducible trajectories.

### 1.4 Per-Trajectory RNG Streams in the Workflow

```python
trajectory_rngs = [np.random.default_rng(s) for s in seed_list]
```

Given, for example:

```python
seed_list = [100, 200, 300]
```

`trajectory_rngs[0]`, `trajectory_rngs[1]`, and `trajectory_rngs[2]` are three independent PRNG streams. Each trajectory therefore has:

- Its own internal RNG state.
- Its own deterministic random sequence.

Trajectory 0 always produces the same rotations, velocity draws, and other stochastic initialization values, provided:

- The same NumPy version is used.
- The same SciPy version is used.
- The same execution order is preserved.

### 1.5 How Random Numbers Are Actually Produced

Consider:

```python
rng.normal(0, 1, size=(N, 3))
```

This does not directly generate Gaussian numbers. Internally:

1. The PRNG generates uniformly distributed bits.
2. These bits are converted into uniform floats in `[0, 1)`.
3. NumPy transforms these uniform values into Gaussian-distributed numbers.

Historically, this transformation has used methods such as Box-Muller, Ziggurat, or inverse transform sampling. Modern NumPy uses optimized, vectorized algorithms for this purpose.

Velocity sampling follows the same principle:

```python
velocities = rng.normal(...)
```

Mathematically, this corresponds to:

```
v_i ~ N(0, σ_i²)
```

with the randomness entirely determined by the RNG state.

### 1.6 Reproducibility of Random Rotations (SciPy Integration)

Given:

```python
R.random(nrot, random_state=rng)
```

from SciPy, SciPy consumes random numbers from the same NumPy `Generator`. Internally, SciPy samples random quaternions (or an equivalent random rotation parameterization) using the RNG stream provided.

As a result, `seed_list[b]` fully determines both orientation and velocity initialization, because both consume numbers from the same deterministic RNG stream.

### 1.7 Order Dependence of RNG Draws

PRNGs are sequential. Consider:

```python
rng.normal()
rng.normal()
```

This consumes two random values. If an additional draw is inserted before them:

```python
rng.uniform()
rng.normal()
rng.normal()
```

then all subsequently generated numbers change.

This is a critical consideration for MD workflows: even with an identical seed, the following can alter all downstream stochastic values:

- Changing code order.
- Adding diagnostics that consume RNG draws.
- Changing batch logic.

### 1.8 Rationale for Separate RNGs per Trajectory

A naive approach uses a single global RNG:

```python
np.random.seed(123)
```

This globally shared RNG is:

- Fragile.
- Order-dependent.
- Difficult to parallelize.

The workflow instead instantiates `trajectory_rngs[b]` per trajectory. Advantages include:

- Deterministic behavior.
- Isolated streams.
- Parallel-safety.
- Reproducible batch behavior.


### 1.9 Reproducibility Across Library Versions

NumPy aims to keep reproducibility stable, but exact streams can change if:

- RNG algorithms change.
- The SciPy rotation implementation changes.
- Floating-point vectorization changes.

Strict bitwise reproducibility is therefore generally only guaranteed with a fixed NumPy version, fixed SciPy version, and fixed hardware/backend. For scientific reproducibility, it is generally necessary to record:

- The seed list.
- The NumPy version.
- The SciPy version.

---

## 2. Seed Parsing, Storage, and Consumption in the MD Workflow

### 2.1 Where the Seed Is Parsed and Stored

**File:** `__init__.py`
**Location:** The "Set per-trajectory seeds" block in `main()`.

Key facts from that block:

- The user-facing key is `general_parameters.seed`.
- If `batch_size > 1` and `seed` is a list/tuple/ndarray → treated as the per-trajectory seed list.
- If `seed` is a comma-separated string (e.g., `"1,2,3"`) → split and treated as the per-trajectory seed list.
- If `batch_size == 1` and `seed` is scalar → used directly.
- If `batch_size > 1` and `seed` is scalar, non-list, or missing → per-trajectory random seeds are generated; a warning is printed if scalar.
- Uniqueness is enforced for batch runs: duplicates in a user-supplied list raise an error; duplicates in a randomly generated list are regenerated.

The resolved per-trajectory seed list is passed to MD initialization:

```python
simulation_parameters["general_parameters"]["seed_list"] = trajectory_seeds
```

The per-trajectory seeds used for stochastic initialization are therefore always stored under `general_parameters.seed_list`.

### 2.2 How the Seed List Is Consumed in MD Initialization

**File:** `md.py`
**Function:** `initialize_collision_simulation`

```python
seed_list_cfg = general_params.get("seed_list", simulation_parameters.get("seed_list", None))
...
trajectory_rngs = [np.random.default_rng(s) for s in seed_list]
```

This creates one NumPy RNG per trajectory, seeded from `seed_list`.

### 2.3 Random Rotation

**File:** `md.py`
**Section:** "random rotation"

```python
coordinates = np.stack(
    [apply_random_rotation(coordinates, n_rotations=None, rng=trajectory_rngs[b])
     for b in range(batch_size)],
    axis=0,
)
```

Each trajectory uses its own RNG, passed into:

**File:** `rotation_utils.py`
**Function:** `apply_random_rotation`

```python
M = R.random(nrot, random_state=rng).as_matrix()
```

**Key fact:** `scipy.spatial.transform.Rotation.random(..., random_state=rng)` is seeded by the per-trajectory RNG passed in from `md.py`.

 Random rotation is therefore fully reproducible per trajectory, based on the seed in `seed_list[b]`.

### 2.4 Initial Velocity Sampling

**File:** `md.py`
**Section:** "Sample velocities"

```python
velocities = np.stack(
    [sample_velocities(species, temperature, rng=trajectory_rngs[b])
     for b in range(batch_size)],
    axis=0,
)
```

`sample_velocities(...)` is defined in:

**File:** `initial_configuration.py`
**Function:** `sample_velocities`

```python
if rng is None:
    rng = np.random
velocities = rng.normal(0.0, 1.0, size=(species.shape[0], 3)) * stddev[:, None]
```

**Key facts:**

- The RNG passed in is trajectory-specific.
- Maxwell–Boltzmann velocity sampling is therefore deterministic per trajectory given the seed.

 Initial velocities are therefore reproducible per trajectory, based on the same `seed_list[b]` that also drives rotation.

### 2.5 Summary: Seed → Rotation + Velocity

```
general_parameters.seed
    → parsed in __init__.py
    → stored as seed_list

seed_list[b]
    → np.random.default_rng(seed_list[b])
    → used for:
        - random rotation       (apply_random_rotation(..., rng=trajectory_rngs[b]))
        - initial velocities    (sample_velocities(..., rng=trajectory_rngs[b]))
```

The seed directly controls both random rotation and velocity initialization per trajectory.

---

## 3. JAX CPU Parallelization

### 3.1 Default Multi-Core Behavior

When `device: cpu` is set, JAX uses all available CPU cores by default for parallel computation. This is why a process such as `pymars` can be observed using 300% CPU — this indicates it is using 3 cores (or threads) at full capacity.

### 3.2 Cause: Multi-Threaded BLAS/LAPACK Libraries

- JAX (and NumPy, SciPy, etc.) rely on multi-threaded BLAS/LAPACK libraries such as OpenBLAS, MKL, or Eigen.
- By default, these libraries use as many threads as there are CPU cores, or as many as allowed by environment variables.
- As a result, a single `pymars` process can use multiple CPU cores, leading to >100% CPU usage as reported by tools such as `top` or `htop`.

### 3.3 Controlling CPU/Thread Usage

The number of threads used can be limited by setting the following environment variables before running `pymars`:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
```

JAX and its dependencies use all available CPU threads by default for performance. These environment variables can be used to restrict CPU usage if desired.

---

## Appendix: File/Function Reference

| File | Function / Section | Role |
|---|---|---|
| `__init__.py` | "Set per-trajectory seeds" block in `main()` | Parses `general_parameters.seed`, resolves and stores `seed_list` |
| `md.py` | `initialize_collision_simulation` | Builds `trajectory_rngs` from `seed_list` |
| `md.py` | "random rotation" section | Applies per-trajectory random rotation using `trajectory_rngs[b]` |
| `md.py` | "Sample velocities" section | Samples per-trajectory initial velocities using `trajectory_rngs[b]` |
| `rotation_utils.py` | `apply_random_rotation` | Calls `scipy.spatial.transform.Rotation.random(..., random_state=rng)` |
| `initial_configuration.py` | `sample_velocities` | Draws Maxwell–Boltzmann-distributed velocities via `rng.normal(...)` |
