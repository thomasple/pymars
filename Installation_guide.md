# PyMARS Installation Guide (CPU and GPU)

## Overview

This guide explains how to install PyMARS and FeNNol on shared HPC clusters and Linux systems. The instructions are designed to:

- Avoid system-wide installations.
- Not require root/admin access.
- Minimize native compilation problems.
- Improve compatibility with older cluster environments.

**Required Python version:** 3.10

**Recommended Python version:** 3.12 or higher (improves by 10%)

**Layout of this guide**

- Section 1: CPU installation.
- Section 2: GPU installation.
- Section 3: notes and troubleshooting.

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
   conda create -n new_env python=3.10 -y
   ```

2. Activate the environment:
   ```bash
   conda activate ~/.conda/envs/new_env
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
python3 -m venv new_env
conda activate new_env
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

### 2.1 Principles

These rules come from problems met in practice (see Section 3 for the details). Follow them on any machine.

1. **One source per package family.** Install the JAX stack (`jax`, `jaxlib`, the CUDA plugin, the CUDA libraries) either entirely with pip or entirely with conda. Mixing the two leaves half-overwritten files (for example a missing `jaxlib/version.py`) and two competing GPU backends.
2. **Pin versions.** Write down the exact versions of `jax`, `jaxlib`, the CUDA plugin, `flax`, `optax` and the NVIDIA libraries once the environment works.
3. **`jax.devices()` is not a GPU test.** It can list the GPU while every computation fails. Always test with a real operation (see 2.3, step 4).
4. **Test on a GPU compute node**, not on the cpu node.
5. **Snapshot a working environment** before changing anything (see 2.3, step 9), and experiment on a clone.
6. **Install secondary packages with `--no-deps`** so that pip cannot replace the JAX stack.

### 2.2 Prerequisites

- NVIDIA GPUs are available on the compute nodes.
- NVIDIA drivers are installed by the cluster administrators (no admin rights are needed for anything below).
- `conda` (or micromamba), `pip`, `git`.

### 2.3 Installation steps

1. **Check the driver, on a GPU node:**
   ```bash
   nvidia-smi
   ```
   The "CUDA Version" shown is the highest CUDA version the driver supports, not what is installed in your environment. For CUDA 12 builds of JAX the driver must be recent enough; check the requirement of the JAX release you choose in its installation notes before proceeding. Also check which modules are loaded (`module list`) and whether `LD_LIBRARY_PATH` points to a CUDA installation that could shadow the libraries in your environment.

2. **Create the environment:**
   ```bash
   conda create -n pymars-env python=3.10 (or the version that you want) -y
   conda activate pymars-env
   pip install --upgrade pip setuptools wheel
   ```
   > **Python version:** the project requires 3.10. Verified also on 3.12, 3.13.

3. **Install CUDA-enabled JAX with pip, with an explicit version:**
   ```bash
   pip install "jax[cuda12]==<VERSION>"
   ```
   Use a version that matches your driver and the rest of the stack (see 2.4). If the install fails because a package called `nvidia-cudnn-cu12` (or another `nvidia-*` package) cannot be built or downloaded, pip has chosen a placeholder release. Pin the library explicitly, for example `pip install "jax[cuda12]==<VERSION>" "nvidia-cudnn-cu12==<VERSION>"`, using the versions listed for your JAX release.

4. **Test immediately, before installing anything else:**
   ```bash
   JAX_PLATFORMS=cuda python -c "import jax, jax.numpy as jnp; print(jax.devices()); jnp.ones(1).block_until_ready(); print('OK')"
   ```
   Expected output: a `CudaDevice(id=0)` entry (older JAX versions print `GpuDevice(id=0)`) followed by `OK`. `JAX_PLATFORMS=cuda` makes JAX print the real error instead of silently falling back to the CPU.

5. **Install `flax` and `optax` without dependencies**, with versions compatible with your JAX (see 2.4):
   ```bash
   pip install --no-deps "flax==<VERSION>" "optax==<VERSION>"
   pip install msgpack rich treescope typing_extensions pyyaml absl-py
   ```

6. **Clone and install PyMARS:**
   ```bash
   git clone https://github.com/thomasple/pymars.git pymars_dir
   cd pymars_dir
   pip install -e . --no-deps
   ```

7. **Install the remaining dependencies:**
   ```bash
   pip install ase numba sympy tomlkit pytest h5py
   ```
   If `h5py` fails to build, install it from conda-forge instead (`conda install -c conda-forge h5py`).

8. **Install FeNNol without dependencies:**
   ```bash
   pip install fennol --no-deps
   ```
   For a development checkout use `pip install --no-deps -e /path/to/FeNNol`. An editable install points to that directory, so every environment installed this way shares the same source code.

9. **Final tests and snapshot:**
   ```bash
   python -c "import jax; print('JAX', jax.__version__, jax.devices())"
   python -c "from fennol import FENNIX; print('FeNNol OK')"
   python -c "import pymars; print('PyMARS OK')"
   pip check
   pymars --help

   mkdir -p ~/env_backup
   conda list --explicit > ~/env_backup/pymars_gpu_conda_explicit.txt
   conda env export > ~/env_backup/pymars_gpu.yml
   pip freeze > ~/env_backup/pymars_gpu_pip.txt
   ```
   `pip check` may report optional dependencies of `flax` (for example `orbax-checkpoint`, `tensorstore`). They are only needed for checkpointing; ignore them if you do not use it.

### 2.4 Version compatibility

- `flax` and `optax` must match the JAX release. A `flax` that is too new fails at import with `cannot import name 'Format' from 'jax.experimental.layout'` (seen with `flax 0.12.x` and `jax 0.4.34`). Upgrading JAX, or pinning an older `flax`/`optax`, solves it.
- The CUDA runtime (the `cuda-version` in conda, or `nvidia-cuda-runtime-cu12` in pip) must be at least the minimum required by the JAX build. For `jax 0.4.34` this is CUDA 12.1.
- cuDNN, cuBLAS, cuFFT, cuSOLVER and cuSPARSE must come from the same source and release family as the runtime. Do not combine them from several channels without testing.
- When something works, pin all of the above (Section 2.3, step 9).

### 2.5 Diagnostics 

| Symptom | Likely cause | What to do |
|---|---|---|
| `jax.devices()` shows only `CpuDevice`, message "a CUDA-enabled jaxlib is not installed" | CPU-only `jaxlib`, or the CUDA plugin is missing | `pip list \| grep -iE "jax\|nvidia"` and `conda list \| grep -iE "jax\|cuda"`; reinstall the JAX stack from a single source |
| `ModuleNotFoundError: No module named 'jaxlib.version'` | `jaxlib` partly overwritten or removed by a mix of pip and conda | Remove `jax*` directories and `*.dist-info` in `site-packages`, then reinstall from one source |
| "Outdated CUDA installation found … minimum supported: 12010" | CUDA runtime older than what this JAX build requires | Use a newer runtime (≥ 12.1 for `jax 0.4.34`) |
| "Unable to load cuBLAS / cuFFT / cuSOLVER / cuSPARSE" | Libraries missing, or an old CUDA found first in the library path | Install the full CUDA library set; check `module list` and `LD_LIBRARY_PATH` |
| `cudaErrorSymbolNotFound` when creating the cuDNN handle, then `DNN library initialization failed` | Mismatch between cuDNN/CUDA libraries and the JAX plugin, or a second CUDA copy loaded | Test on a clone with different library versions; list the loaded libraries (see below) |
| `jax.devices()` lists the GPU but computations fail | Same as the row above; `jax.devices()` does not initialize cuDNN | Always test with `jnp.ones(1).block_until_ready()` |
| `cannot import name 'Format' from 'jax.experimental.layout'` | `flax` too new for this JAX | Pin an older `flax`/`optax`, or upgrade JAX |
| `scipy._cyutility does not export expected C function` | `scipy` files from two installs mixed (pip and conda) | Remove the pip copy, then `conda install --force-reinstall scipy numpy` |
| pip fails to build metadata for `tensorstore` | Old OS, no wheel; pulled in by `flax`/`orbax` | Install `flax`/`optax` with `--no-deps` |
| pip fails on a placeholder `nvidia-cudnn-cu12` | pip picked a placeholder release | Pin the real version explicitly |

To see which CUDA libraries a Python process actually loads:
```bash
python - <<'EOF'
import re, jax, jax.numpy as jnp
jnp.ones(1).block_until_ready()
libs = set()
for line in open('/proc/self/maps'):
    m = re.search(r'(/\S*(cudnn|cublas|cudart|xla_cuda)\S*)', line)
    if m:
        libs.add(m.group(1))
print("\n".join(sorted(libs)))
EOF
```
Paths from more than one location (the environment plus a cluster module, for example) mean two copies are in conflict.

### 2.6 Runtime options

Force CUDA backend (shows real errors instead of falling back to the CPU):
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

Device selection in `input.yaml`:
```yaml
calculation_parameters:
  device: cuda:0
```

### 2.7 Minimal fallback installation (GPU)

```bash
conda create -n pymars-env python=3.10 -y
conda activate pymars-env
pip install --upgrade pip setuptools wheel
pip install "jax[cuda12]==<VERSION>"
python -c "import jax, jax.numpy as jnp; jnp.ones(1).block_until_ready(); print(jax.devices(), 'OK')"

git clone https://github.com/thomasple/pymars.git pymars_new
cd pymars_new
pip install -e . --no-deps
pip install --no-deps "flax==<VERSION>" "optax==<VERSION>"
pip install pyyaml numba sympy msgpack rich treescope typing_extensions absl-py ase tomlkit h5py
pip install fennol --no-deps

python -c "import jax; print(jax.devices())"
```

If you can, avoid building an environment from scratch: clone or export one that already works (Section 3.3).

---

## Section 3 — Troubleshooting notes

Valid in particular on older versions machines, or not updated.

### 3.1 Machine profile example

| Item | Value | Status |
|---|---|---|
| OS / kernel | Linux 3.10.0-1160.88.1.el7.x86_64 (Enterprise Linux 7 family) | 
| GPU driver | NVIDIA 525.89.02 ("CUDA Version: 12.0" in `nvidia-smi`) |
| Default compiler | gcc 4.8.5, no `g++` available | 
| System glibc | expected to be 2.17 (typical for this OS) |; 
| Nodes with several GPUs | at least one node shows `CudaDevice(id=0)` and `CudaDevice(id=1)` |

The old system compiler cannot build NumPy 2.x or other recent packages from source: use binary wheels or conda packages, never source builds.

### 3.2 Reference environment 

The environment `test_environment` Main contents:

| Component | Version | Source |
|---|---|---|
| Python | 3.13.9 | conda-forge |
| jax, jaxlib | 0.5.3 | pip |
| jax-cuda12-plugin, jax-cuda12-pjrt | 0.5.3 | pip |
| CUDA toolkit (`cuda-version`) | 12.9 | conda, channel `nvidia` |
| nvidia-cudnn-cu12 | 9.5.0.50 | pip |
| other `nvidia-*-cu12` | cublas 12.6.4.1, cufft 11.4.1.4, cusolver 11.7.1.2, cusparse 12.5.10.65, nccl 2.27.7, runtime 12.9.79, nvjitlink 12.9.86 | pip |
| flax / optax | 0.10.6 / 0.2.6 | pip |
| numpy / numba | 2.3.5 / 0.63.1 | pip |
| scipy | 1.16.0 | conda |
| FeNNol | 2025.12.12.post1, editable from `/home/user/FeNNol` | pip |

This environment has both the conda `nvidia`-channel CUDA toolkit and the pip `nvidia-*-cu12` libraries. It works, but it is not known whether both are needed. 

### 3.3 Reproducing the environment

**Preferred path: clone.** 
If you have an already working environment, clone it.
Check disk space first (the CUDA toolkit adds several GB) and run the clone in a job or with `nohup`, because it can take a long time and conda prints no progress:
```bash
nohup conda create -n new_env --clone work_env > clone.log 2>&1 &
```
Then reinstall editable packages that must not share sources:
```bash
pip install --no-deps -e /home/user/pymars_dev
```
Check that entry points use the new environment: `head -1 $(which pymars)`.

**Export a working environment** (do it after every successful change):
```bash
conda list --explicit > new_env_conda_explicit.txt
conda env export -n new_env > new_env.yml
pip freeze > new_env_pip.txt
```
Editable installs (lines starting with `-e` or containing `@ /home/...`) are not reproduced from a version list; reinstall them with `pip install --no-deps -e <directory>`.

**From scratch (painful):**
```bash
conda create -n new_env python=3.13 (or any python version >=3.10)
conda activate new_env
conda install -c nvidia -c conda-forge "cuda-version=12.9" "cuda-toolkit=12.9.1" scipy=1.16
pip install numpy==2.3.5 numba==0.63.1 jax==0.5.3 jaxlib==0.5.3 \
    jax-cuda12-plugin==0.5.3 jax-cuda12-pjrt==0.5.3 flax==0.10.6 optax==0.2.6 \
    nvidia-cudnn-cu12==9.5.0.50
pip install --no-deps -e /home/user/FeNNol
```
Test it with the command in Section 2.3, step 4.

### 3.4 Possible problems 

1. **`numpy==2.4.6` fails to build** (`Unknown compiler(s): c++, g++ …`; gcc 4.8.5). No compatible wheel was found, so pip fell back to a source build. Fix: install NumPy from conda-forge, or pin a version for which wheels install (`pip install --only-binary=:all: <package>` fails fast instead of building).
2. **`conda install jax[cuda12]` does not work.** `[cuda12]` is a pip extra. Use pip for `jax[cuda12]`, or conda packages with a CUDA build string.
3. **`cuda-version=12.0` is too old** for `jax 0.4.34` ("Version JAX was built against: 12000, minimum supported: 12010"). CPU fallback message: "a CUDA-enabled jaxlib is not installed".
4. **Mixed pip/conda JAX stack.** After removing pip packages, `jaxlib/version.py` was missing (`No module named 'jaxlib.version'`). Reinstalling `jaxlib` with conda (`--force-reinstall`) fixed the import. Removing `jax-cuda12-plugin` and `jax-cuda12-pjrt` from an environment that relied on them made the GPU disappear again: they provide the GPU backend for `jax 0.4.x` when installed.
5. **pip failed on `nvidia-cudnn-cu12 9.27.0.42.dev…`** (placeholder package that needs `pypi.nvidia.com`). Pinning the real version, `9.5.0.50`, is what the working environment has.
6. **`tensorstore` metadata build failure** when pip pulls it in through `flax`/`orbax`. Avoid with `--no-deps`.
7. **`flax 0.12.10` with `jax 0.4.34`:** `cannot import name 'Format'`. FeNNol requires JAX/flax but does not pin versions. `flax 0.10.6` with `jax 0.5.3` works.
8. **`scipy._cyutility does not export expected C function slice_memviewslice`** after installing `numba` from conda-forge on top of a mixed environment. Reinstall `scipy` and `numpy` from a single source.
9. **`cudaErrorSymbolNotFound` / `DNN library initialization failed`** in `environment`: `jax 0.4.34` (conda `jaxlib` cuda126 plus pip plugin 0.4.34), `cuda-version 12.6`, `cudnn 9.3.0.75` from conda-forge. `jax.devices()` listed the GPU but the first computation failed. **Cause not conclusively identified.** The differences from the working environment are the JAX release (0.4.34 vs 0.5.3), cuDNN (9.3 vs 9.5), the CUDA release (12.6 vs 12.9) and a mixed pip/conda GPU stack. The failing environment was replaced by the clone of the working one.

### 3.5 Checklist

- Test on a GPU node with `jnp.ones(1).block_until_ready()`, not only `jax.devices()`.
- Do not run `pip install -U jax`, `flax` or `numpy` in the working environment; test changes on a clone.
- Before installing with conda, preview with `conda install --dry-run`. Stop if it wants to change `jax`, `jaxlib`, `cuda-version` or `cudnn`.
- Install secondary packages with `pip install --no-deps`.
- Do not compile from source (gcc 4.8.5); prefer wheels (`--only-binary=:all:`) or conda packages.
- Update job scripts (`conda activate …`) when an environment is renamed or removed (`grep -rIl "<env_name>" ~/ --include="*.sh" --include="*.pbs"`).
- `conda clean --all` only removes caches and is safe for environments located in `miniconda3/envs/`.
