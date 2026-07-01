import os
import numpy as np

def read_xyz_file(xyz_path: str) -> tuple[list[str], np.ndarray]:
    """Read a single-frame XYZ file."""
    if not os.path.exists(xyz_path):
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    species = []
    coords = []

    with open(xyz_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"XYZ file {xyz_path} has fewer than 2 lines")

    try:
        n_atoms = int(lines[0].strip())
        #print(f"Number of atoms: {n_atoms}")
    except ValueError:
        raise ValueError(f"First line must be an integer (number of atoms)")

    if len(lines) < 2 + n_atoms:
        raise ValueError(f"XYZ file claims {n_atoms} atoms but has only {len(lines) - 2} data lines")

    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Line {i+1} has fewer than 4 fields")
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"Line {i+1} has non-numeric coordinates")
        species.append(sym)
        coords.append([x, y, z])

    return species, np.array(coords, dtype=np.float64)