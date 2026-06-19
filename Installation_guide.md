# PyMARS Installation Guide (CPU and GPU)

## Overview

This guide explains how to install PyMARS and FeNNol on shared HPC clusters and Linux systems. The instructions are designed to:

- Avoid system-wide installations.
- Not require root/admin access.
- Minimize native compilation problems.
- Improve compatibility with older cluster environments.

**Required Python version:** 3.10

---

## Section 1 — CPU Installation

### 1.1 Prerequisites

The following must be installed beforehand:

- conda
- Python
- pip
- git
- uv (if needed, install with `pip install uv`)
- GCC ≥ 11.2

> **Important:** GCC must be version 11.2 or higher. On older cluster nodes, load the most recent GCC module available.

### 1.2 Installation Steps

1. Create a virtual environment:
   ```bash
   conda create -n pymars-env python=3.10 -y
   ```

2. Activate the environment:
   ```bash
   conda activate ~/.conda/envs/pymars-env
   ```

3. Upgrade build tools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. Install CPU JAX:
   ```bash
   pip install --upgrade jax
   ```

5. Fetch PyMARS (clone, or navigate to an existing program directory):
   ```bash
   git clone https://github.com/thomasple/pymars.git pymars_dir
   cd pymars_dir
   ```

6. Generate/update the uv lockfile:
   ```bash
   uv lock --upgrade
   ```

7. Install dependencies from the project TOML file:
   ```bash
   pip install .
   ```
   If editable mode is preferred:
   ```bash
   pip install -e . --no-deps
   ```

8. If something did not install, check the GCC version, then install remaining dependencies manually.

   Unproblematic dependencies:
   ```bash
   pip install ase numba sympy tomlkit pytest msgpack rich treescope pyyaml typing_extensions
   ```

   GCC-sensitive dependencies:
   ```bash
   pip install \
       h5py \
       optax \
       flax
   ```

9. Install FeNNol:
   ```bash
   pip install fennol --no-deps
   ```

10. Test the installation:
    ```bash
    python -c "import jax; print('JAX version:', jax.__version__)"
    python -c "import jax; print('Devices:', jax.devices())"
    python -c "from fennol import FENNIX; print('FeNNol OK')"
    python -c "import pymars; print('PyMARS OK')"
    pymars --help
    ```

### 1.3 Common Problems (CPU)

**1. `h5py` build failure**

Typical error:
```
libhdf5.so: cannot open shared object file
```
Cause:
- pip attempted local compilation.
- HDF5 development libraries are unavailable.

Fixes:
- Avoid local compilation.
- Prefer precompiled wheels.

**2. `tensorstore` build failure**

Typical errors:
```
GLIBC_2.25 not found
GLIBCXX_* not found
CXXABI_* not found
```
Cause:
- Cluster operating system is too old.
- Incompatible compiled binaries.

Fixes:
- Avoid optional `tensorstore`/`orbax` dependencies.
- Avoid `flax[all]`.
- Install lightweight versions without dependencies:
  ```bash
  pip install flax --no-deps
  pip install optax --no-deps
  pip install rich
  ```
- Ensure `pyyaml` is installed.
- Then run the program. If `absl` is reported missing but `flax`, `optax`, and `rich` are present:
  ```bash
  pip cache purge
  pip install optax rich
  ```
  (`absl` is a dependency of `optax`/`rich`, so installing them with dependencies resolves this.)

**Troubleshooting sequence for `flax`/`optax`/`rich` issues:**
```bash
# Install h5py and other problematic packages from conda-forge
conda install -c conda-forge h5py tensorstore -y

# Skip FeNNol and just install pymars with minimal dependencies
pip install -e . --no-deps
pip install pyyaml ase numpy scipy

# Install flax WITHOUT the [all] extras (avoids tensorstore rebuild)
pip install flax --no-deps
pip install msgpack rich treescope typing_extensions

# Now install fennol without dependencies
pip install fennol --no-deps

# Install remaining fennol dependencies manually
pip install numba sympy tomlkit pytest
pip install optax pyyaml

# Finally install pymars
cd ~/pymars_new
pip install -e . --no-deps
pip install pyyaml ase
```
Alternatively, micromamba packages can be used instead of pip.

**3. AVX runtime errors**

Cause:
- Login nodes may use older CPUs incompatible with JAX wheels.

Fix:
- Run on compute nodes instead.

**4. Disk space issues**

Fix:
```bash
pip cache purge
```

### 1.4 Minimal Fallback Installation (CPU)

```bash
python3 -m venv pymars-env
source pymars-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install uv
pip install jax
git clone https://github.com/thomasple/pymars.git pymars_new
cd pymars_new
uv lock --upgrade
pip install -e . --no-deps
pip install \
    flax --no-deps \
    optax --no-deps \
    rich --no-deps
pip install \
    pyyaml \
    numba \
    sympy \
    msgpack \
    ase
pip install fennol --no-deps
```

If installation still fails:
- Avoid optional extras.
- Avoid `tensorstore`/`orbax`.
- Use micromamba instead of pure pip.
- Run on compute nodes rather than login nodes.

### 1.5 Module Note: GCC ≥ 11.2 (Important)

```bash
pip install jax
pip install flax --no-deps
pip install optax   # pulls in absl-py
```

Work around failing dependencies by installing with `--no-deps`. Do not attempt to update `tensorstore`, `h5py`, or `hdf5` if the first attempt fails — this is most likely a waste of time.

---

## Section 2 — GPU Installation

### 2.1 Prerequisites

This installation assumes:

- NVIDIA GPUs are available.
- CUDA drivers are already installed by the cluster administrators.
- The user has no admin privileges.

### 2.2 Installation Steps

1. Create a virtual environment:
   ```bash
   python3.10 -m venv pymars-gpu
   ```

2. Activate the environment:
   ```bash
   source pymars-gpu/bin/activate
   ```

3. Upgrade build tools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. Install uv:
   ```bash
   pip install uv
   ```

5. Verify GPU visibility:
   ```bash
   nvidia-smi
   ```

6. Check the CUDA version (also via `nvidia-smi`), then install CUDA-enabled JAX accordingly.

   For CUDA 12:
   ```bash
   pip install --upgrade "jax[cuda12]"
   ```

   For CUDA 11:
   ```bash
   pip install --upgrade "jax[cuda11]"
   ```

   Alternative installation method:
   ```bash
   python -m pip install \
     --upgrade \
     "jax[cuda12]" \
     -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

7. Clone PyMARS:
   ```bash
   git clone https://github.com/thomasple/pymars.git pymars_new
   cd pymars_new
   ```

8. Generate/update the uv lockfile:
   ```bash
   uv lock --upgrade
   ```

9. Install from the project TOML file:
   ```bash
   pip install .
   ```
   If editable mode is preferred:
   ```bash
   pip install -e . --no-deps
   ```

10. Install remaining dependencies:
    ```bash
    pip install \
        h5py \
        ase \
        numba \
        optax \
        sympy \
        tomlkit \
        pytest \
        flax \
        msgpack \
        rich \
        treescope \
        typing_extensions \
        pyyaml
    ```

11. Install FeNNol:
    ```bash
    pip install fennol --no-deps
    ```

12. Test GPU visibility:
    ```bash
    python -c "import jax; print(jax.devices())"
    ```
    Expected output:
    ```
    [GpuDevice(id=0)]
    ```

### 2.3 Common Problems (GPU)

**1. GPU not detected**

Typical output:
```
Devices: [CpuDevice(id=0)]
```
Fix:
```bash
python -m pip uninstall -y jax jaxlib
```
Then reinstall CUDA-enabled JAX.

**2. CUDA mismatch**

Cause:
- The installed JAX wheel is incompatible with the CUDA runtime.

Fix:
- Verify the CUDA version:
  ```bash
  nvidia-smi
  ```
- Reinstall the matching JAX wheel.

**3. GLIBC / tensorstore / flax errors**

Typical errors:
```
GLIBC_2.25 not found
GLIBCXX_* not found
```
Fix:
```bash
pip install flax --no-deps
pip install optax --no-deps
pip install rich --no-deps
```
Avoid:
- `flax[all]`
- `orbax`
- tensorstore-heavy extras

**4. AVX runtime errors**

Cause:
- Old CPUs on login nodes.

Fix:
- Use compute/GPU nodes only.

### 2.4 Troubleshooting (GPU)

Force CUDA backend:
```bash
export JAX_PLATFORMS=cuda
```

Select GPU manually:
```bash
export MARS_DEVICE=cuda:0
```

Force CPU mode:
```bash
export MARS_DEVICE=cpu
```

### 2.5 Minimal Fallback Installation (GPU)

```bash
python3 -m venv pymars-gpu
source pymars-gpu/bin/activate
pip install --upgrade pip setuptools wheel
pip install uv
pip install --upgrade "jax[cuda12]"
git clone https://github.com/thomasple/pymars.git pymars_new
cd pymars_new
uv lock --upgrade
pip install -e . --no-deps
pip install \
    flax --no-deps \
    optax --no-deps \
    rich --no-deps
pip install \
    pyyaml \
    numba \
    sympy \
    msgpack \
    ase
pip install fennol --no-deps
```

Test:
```bash
python -c "import jax; print(jax.devices())"
```
Expected output:
```
[GpuDevice(id=0)]
```

### 2.6 Additional Notes

Device selection in `input.yaml`:
```yaml
calculation_parameters:
  device: cuda:0
```

Or via environment variable:
```bash
export MARS_DEVICE=cuda:0
```
