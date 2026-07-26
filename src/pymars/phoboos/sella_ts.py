import numpy as np
from pathlib import Path
import os

from ase.io import read, write
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



def run_tsopt(xyz_file,  model_file, log_file, outfile=None, total_charge=0, fmax=0.05, max_steps=1000):
    "Optimize the molecular geometry of a transition state using a FENNIX model and Sella optimizer(RS-PRFO)"

    # read the coordinates from the xyz file
    print(f"Reading coordinates from: {xyz_file}")
    symbols, coordinates, comment = last_xyz_frame(
        xyz_file)
    print(f"Read {len(symbols)} atoms.")
    #coordinates = np.array(coordinates)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)
    nat = len(species)
    if total_charge != 0:
        print(f"Using total charge of {total_charge} for the system.")
    inputs = {
        "species": species,
        "natoms": np.array([nat], dtype=np.int32),
        "batch_index": np.array([0] * nat, dtype=np.int32),
        "total_charge": np.array([total_charge], dtype=np.int32),
    }

    xyz_filepath = Path(xyz_file)

    # Load the FENNIX model
    model_file = Path(model_file)
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=False)
    #convert = au.KCALPERMOL / model.Ha_to_model_energy 

    # Optimize the geometry using Sella optimizer (RS-PRFO)
    #print("Setup transition state optimization...")
    atoms= read(xyz_filepath)
    atoms.calc = FENNIXCalculator(model, inputs=inputs, gpu_preprocessing=True)
    output_file = Path(outfile) if outfile else xyz_filepath.with_suffix(".tsopt.xyz")
    bintraj_file = xyz_filepath.with_suffix(".trj")
    traj_file = xyz_filepath.with_suffix(".traj.xyz")
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
    print("#######################################################")
    print(f"Final energy: {atoms.get_potential_energy():.6f} eV")

    write(f"{str(output_file)}", atoms)

    print(f"Transition state optimization completed. Final geometry saved to {output_file}.")
    print("#######################################################")

def run_tssrc(xyz_file1, xyz_file2, model_file, log_file, n_images=12, outfile=None, total_charge=0):
    # Set up the NEB calculation for transition state search between two geometries
    # Setting up inputs for the FENNIX model
    print(f"Reading coordinates from: {xyz_file}")
    symbols, coordinates, comment = last_xyz_frame(
        xyz_file)
    print(f"Read {len(symbols)} atoms.")
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)
    nat = len(species)
    if total_charge != 0:
        print(f"Using total charge of {total_charge} for the system.")
    inputs = {
        "species": species,
        "natoms": np.array([nat], dtype=np.int32),
        "batch_index": np.array([0] * nat, dtype=np.int32),
        "total_charge": np.array([total_charge], dtype=np.int32),
    }
    
    # Load the FENNIX model
    model_file = Path(model_file)
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=False)
    
    # Nudged Elastic Band (NEB) sequence
    # 1. Load endpoints
    xyz_file1 = read(xyz_file1)
    xyz_file2 = read(xyz_file2)
    print(f"Loading inputs complete. Starting NEB optimization with {n_images} images.")
    # 2. Create images (copies of reactant to be interpolated)
    images = [xyz_file1] + [xyz_file1.copy() for _ in range(n_images)] + [xyz_file2]
    
    # 3. Attach FENNIX calculator to each image
    for image in images[1:-1]:
        image.calc = FENNIXCalculator(model, inputs=inputs, gpu_preprocessing=True)
    
    # 4. Create the NEB object and interpolate
    neb = NEB(images, climb=True)   # climb=True activates CI-NEB
    neb.interpolate()               # linear interpolation by default

    # 5. Optimize the band
    opt = LBFGS(neb, trajectory="neb.trj", logfile=log_file)
    print("Starting NEB optimization...")
    opt.run(fmax=0.05, steps=500)

    # 6. Find the highest-energy image
    print("NEB optimization complete. Analyzing energies to find transition state guess...")
    energies = [image.get_potential_energy() for image in images[1:-1]]
    peak_index = energies.index(max(energies)) + 1  # +1 because images[0] is reactant
    ts_guess = images[peak_index]
    ts_window = images[peak_index-2:peak_index+2]  # Get 4 images around the peak for refinement
    # 7. Save the transition state guess to an XYZ file
    guess_file = Path("".join(os.path.basename(log_file)),"ts_guess.xyz")
    print(f"Transition state guess saved to: {guess_file}")
    write(str(guess_file), ts_guess)

    return ts_guess, guess_file, ts_window, peak_index