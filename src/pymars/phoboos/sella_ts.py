import numpy as np
from pathlib import Path
import os

from ase.atoms import Atoms
from ase.io import read, write
from ase.build import minimize_rotation_and_translation

try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB
from ase.optimize import LBFGS
from fennol.ase import FENNIXCalculator
from sella import Sella

from fennol.models import FENNIX
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX
from fennol.utils.atomic_units import au
from fennol.utils.io import last_xyz_frame

from pymars.utils import write_xyz_frame, format_batch_conformations, us

def _coerce_atoms(endpoint):
    """Return an ASE Atoms object from either a file path or an Atoms instance."""
    if isinstance(endpoint, Atoms):
        return endpoint.copy()
    return read(endpoint)


def run_tsopt(xyz_file,  model_file, log_file, outfile=None, total_charge=0, fmax=0.05, max_steps=1000):
    "Optimize the molecular geometry of a transition state using a FENNIX model and Sella optimizer(RS-PRFO)"

    atoms = _coerce_atoms(xyz_file)
    if isinstance(xyz_file, Atoms):
        print(f"Using ASE Atoms object with {len(atoms)} atoms for TS optimization.")
    else:
        print(f"Reading coordinates from: {xyz_file}")
        print(f"Read {len(atoms)} atoms.")
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in atoms.get_chemical_symbols()], dtype=np.int32)
    nat = len(species)

    #Setting up total charge for the system
    if total_charge != 0:
        print(f"Using total charge of {total_charge} for the system.")
    inputs = {
        "species": species,
        "natoms": np.array([nat], dtype=np.int32),
        "batch_index": np.array([0] * nat, dtype=np.int32),
        "total_charge": np.array([total_charge], dtype=np.int32),
    }
    charges = np.zeros(nat); charges[0] = total_charge
    atoms.set_initial_charges(charges)

    # Load the FENNIX model
    model_file = Path(model_file)
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=False)
    #convert = au.KCALPERMOL / model.Ha_to_model_energy 

    # Optimize the geometry using Sella optimizer (RS-PRFO)
    #print("Setup transition state optimization...")
    atoms.calc = FENNIXCalculator(model, inputs=inputs, gpu_preprocessing=True)
    if outfile:
        output_file = Path(outfile)
    elif isinstance(xyz_file, Atoms):
        output_file = Path(log_file).with_name("tsopt.xyz")
    else:
        output_file = Path(xyz_file).with_suffix(".tsopt.xyz")
    bintraj_file = output_file.with_suffix(".trj")
    traj_file = output_file.with_suffix(".traj.xyz")
    print(f"Transition state optimization trajectory will be saved to: {traj_file}")
    print("Initialization complete.")
    # Hessian-based saddle point optimization (default Sella behavior)
    opt = Sella(
            atoms,
            order=1,           # 1 = TS (saddle point), 0 = minimum
            trajectory=str(bintraj_file),
            logfile=str(log_file),
        )
    
    print("Starting transition state optimization...")
    opt.run(fmax=fmax, steps=max_steps)

    # convert .trj to .xyz
    traj_frames = read(str(bintraj_file), index=":")  # ":" reads all frames
    write(str(traj_file), traj_frames)

    # delete the binary .trj
    bintraj_file.unlink()

    print("\n")
    print("################################################################################")
    print(f"Final energy: {atoms.get_potential_energy():.6f} eV")

    write(f"{str(output_file)}", atoms)

    print(f"Transition state optimization completed. Final geometry saved to {output_file}.")
    print("################################################################################")

    return atoms, output_file


def run_tssrc(xyz_file1, xyz_file2, model_file, log_file, n_images=12, outfile=None, total_charge=0, max_steps=1000, mode="loose"):
    "Run a transition state search using Nudged Elastic Band (NEB) method with a FENNIX model"
    " the algorithm ues the NEB-TS approach to find a transition state guess between two endpoints (reactant and product)."
    " In particular, it tries to  loosely optimize every single image along the path to the minimum energy path (MEP)"
    "   and loosens the climbing image to a medium tolerance )) "
    " Once this loose threshold is reached, it terminates the NEB phase early and hands the climbing image's coordinates directly to "
    " a formal analytical Transition State optimizer (OptTS), which tightly converges to a final tolerance"
    #fmax in eV/Angstrom
    
    # Setting up inputs for the FENNIX model
    atoms1 = _coerce_atoms(xyz_file1)
    atoms2 = _coerce_atoms(xyz_file2)

    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in atoms1.get_chemical_symbols()], dtype=np.int32)
    nat = len(species)

    #Setting up total charge for the system
    if total_charge != 0:
        print(f"Using total charge of {total_charge} for the system.")
    inputs = {
        "species": species,
        "natoms": np.array([nat], dtype=np.int32),
        "batch_index": np.array([0] * nat, dtype=np.int32),
        "total_charge": np.array([total_charge], dtype=np.int32),
    }
    charges = np.zeros(nat); charges[0] = total_charge
    atoms1.set_initial_charges(charges)
    atoms2.set_initial_charges(charges)

    # Load the FENNIX model
    model_file = Path(model_file)
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=False)

    # Nudged Elastic Band (NEB) sequence
    # 1. Setup endpoints for NEB
    # Attach FENNIX calculator to the endpoints
    atoms1.calc = FENNIXCalculator(model, inputs=inputs, gpu_preprocessing=True)
    atoms2.calc = FENNIXCalculator(model, inputs=inputs, gpu_preprocessing=True)
    
    #Align product onto reactant before creating images
    minimize_rotation_and_translation(atoms1, atoms2)
    print(f"Loading and aligning inputs complete. Starting NEB optimization with {n_images} images.")

    # 2. Create images (copies of reactant to be interpolated)
    images = [atoms1] + [atoms1.copy() for _ in range(n_images)] + [atoms2]
    # 3. Attach FENNIX calculator to each image
    for image in images[1:-1]:
        image.set_initial_charges(charges)
        image.calc = FENNIXCalculator(model, inputs=inputs, gpu_preprocessing=True)
    
    # 4. Create the NEB images and start the search for the transition state guess
    # stage 1: loose band relaxation
    neb = NEB(images, method='improvedtangent', climb=False)
    neb.interpolate('idpp')  # IDPP:Image Dependent Pair Potential interpolation, essential with topologically different endpoints
    binneb_file = Path(log_file).with_name("neb.trj")
    traj_xyz_file = Path(log_file).with_name("neb.traj.xyz")

    opt = LBFGS(neb, trajectory=str(binneb_file), logfile=log_file)
    print("Starting NEB optimization (stage 1: images relaxation)...")
    opt.run(fmax=0.1, steps=max_steps)

    # 5. Activate climbing image and new instance of LBFGS for climbing image optimization
    # stage 2: same band, CI on, medium tolerance, fresh optimizer, smaller steps
    print("NEB optimization stage 2: climbing image...")
    neb.climb = True
    opt = LBFGS(neb, trajectory=str(binneb_file), logfile=str(log_file), maxstep=0.1)
    opt.run(fmax=0.05, steps=max_steps)

    # 6. Find the highest-energy image
    print("Image relaxation complete. Analyzing energies to find highest energy image...")
    energies = [image.get_potential_energy() for image in images]
    peak_index = energies.index(max(energies))
    print(f"Highest energy image found at index {peak_index} with energy {energies[peak_index-1]:.6f} eV.")

    #6B For a zoom refinement, using the immediate neighbors of the peak image as new endpoints and run a new NEB search with closer images
    if mode == "tight":
        print("Refinement mode: using immediate neighbors of the peak image as new endpoints for a tighter NEB search.")
        lo, hi = images[peak_index-1].copy(), images[peak_index+1].copy()
        zooms = [lo] + [lo.copy() for _ in range(n_images//2)] + [hi]
        # Attach FENNIX calculator to each image
        for zoom in zooms:
            zoom.set_initial_charges(charges)
            zoom.calc = FENNIXCalculator(model, inputs=inputs, gpu_preprocessing=True)
        neb2 = NEB(zooms, method='improvedtangent', climb=True)
        neb2.interpolate('idpp')
        opt2 = LBFGS(neb2, trajectory=str(binneb_file), logfile=str(log_file), maxstep=0.1)
        print("NEB optimization stage 3: zoom refinement...")
        opt2.run(fmax=0.05, steps=max_steps)

        energies2 = [im.get_potential_energy() for im in zooms]
        peak2 = int(np.argmax(energies2[1:-1])) + 1
        ts_guess = zooms[peak2].copy()

    if mode == "loose":
        ts_guess = images[peak_index].copy()

    #7. convert and clean up trajectory
    neb_frames = read(str(binneb_file), index=":")
    write(str(traj_xyz_file), neb_frames)
    binneb_file.unlink()

    # 7. Save the transition state guess to an XYZ file
    guess_file = Path(outfile) if outfile else Path(log_file).with_name("ts_guess.xyz")
    print(f"Transition state guess saved to: {guess_file}")
    write(str(guess_file), ts_guess)

    return ts_guess, guess_file
