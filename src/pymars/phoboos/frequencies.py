import numpy as np
from pathlib import Path

from ase.atoms import Atoms
from ase.io import read
from ase.vibrations import Vibrations

from fennol.ase import FENNIXCalculator
from fennol.models import FENNIX
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX
from pymars.utils import us

def _coerce_atoms(endpoint):
    """Return an ASE Atoms object from either a file path or an Atoms instance."""
    if isinstance(endpoint, Atoms):
        return endpoint.copy()
    return read(endpoint)


def _reshape_mode(mode, natoms):
    """Reshape a vibrational mode to (natoms, 3) if necessary."""
    mode = np.asarray(mode, dtype=float)
    if mode.shape == (natoms * 3,):
        mode = mode.reshape(natoms, 3)
    elif mode.shape != (natoms, 3):
        mode = mode.reshape(natoms, 3)
    return mode


def _estimate_reduced_mass(mode, masses):
    """Estimate the reduced mass (in amu) for a vibrational mode."""
    mode = np.asarray(mode, dtype=float)
    norm = np.linalg.norm(mode)
    if norm == 0.0:
        return np.nan
    normalized = mode / norm
    denominator = np.sum((normalized ** 2) / masses[:, None])
    if denominator == 0.0:
        return np.nan
    return 1.0 / denominator


def _force_constant_from_frequency(freq_cm, reduced_mass_amu):
    """Estimate the force constant (in N/m) from the vibrational frequency (in cm^-1) and reduced mass (in amu)."""
    if not np.isfinite(freq_cm) or not np.isfinite(reduced_mass_amu):
        return np.nan
    amu_to_kg = 1.66053906660e-27
    speed_of_light_cm_s = 2.99792458e10
    omega = 2.0 * np.pi * speed_of_light_cm_s * abs(freq_cm)
    force_constant_n_m = reduced_mass_amu * amu_to_kg * omega**2
    return force_constant_n_m * 0.01


def _format_frequency(freq):
    """Return a real-valued wavenumber, preserving imaginary modes as negative values."""
    freq = np.asarray(freq)
    if np.iscomplexobj(freq):
        real_part = float(np.real(freq))
        imag_part = float(np.imag(freq))
        if np.isclose(imag_part, 0.0):
            return real_part
        if np.isclose(real_part, 0.0):
            return -abs(imag_part)
        return real_part if abs(real_part) >= abs(imag_part) else -abs(imag_part)
    return float(freq)


def _format_report(atoms, vib, energy_conv):
    "Format the vibrational analysis report for output."
    freqs = np.asarray([_format_frequency(freq) for freq in vib.get_frequencies()], dtype=float)
    energies = np.asarray(vib.get_energies(), dtype=float)
    energies_kcalmol = energies * energy_conv
    masses = np.asarray(atoms.get_masses(), dtype=float)
    atomic_numbers = atoms.get_atomic_numbers()
    natoms = len(atoms)

    modes = []
    reduced_masses = []
    force_constants = []
    for index, freq in enumerate(freqs):
        mode = _reshape_mode(vib.get_mode(index), natoms)
        modes.append(mode)
        reduced_mass = _estimate_reduced_mass(mode, masses)
        reduced_masses.append(reduced_mass)
        force_constants.append(_force_constant_from_frequency(freq, reduced_mass))

    lines = []
    lines.append("")
    lines.append(" Harmonic vibrational analysis")
    lines.append("")

    for start in range(0, len(freqs), 3):
        block = list(range(start, min(start + 3, len(freqs))))
        lines.append("".join(f"{mode_index + 1:>22d}" for mode_index in block))
        lines.append("".join(f"{'A':>22s}" for _ in block))
        lines.append(" Frequencies --" + "".join(f"{freqs[mode_index]:>22.4f}" for mode_index in block))
        lines.append(" Red. masses --" + "".join(f"{reduced_masses[mode_index]:>22.4f}" for mode_index in block))
        lines.append(" Frc consts  --" + "".join(f"{force_constants[mode_index]:>22.4f}" for mode_index in block))
        mode_header = ["  Atom  AN"]
        for _ in block:
            mode_header.append("      X      Y      Z")
        lines.append("".join(mode_header))
        for atom_index in range(natoms):
            row = f"{atom_index + 1:6d}{atomic_numbers[atom_index]:4d}"
            for mode_index in block:
                x, y, z = modes[mode_index][atom_index]
                row += f"{x:9.2f}{y:7.2f}{z:7.2f}"
            lines.append(row)
        lines.append("")

    lines.append(" Vibrational mode energies")
    lines.append("  Mode       eV           kcal/mol")
    for index, (energy_ev, energy_kcalmol) in enumerate(zip(energies, energies_kcalmol), start=1):
        lines.append(f"{index:6d}{energy_ev:13.6f}{energy_kcalmol:15.6f}")

    lines.append("")
    lines.append(" Vibrational frequencies:")
    for index, freq in enumerate(freqs, start=1):
        lines.append(f"{index:6d}: {freq:12.4f} cm^-1")

    return "\n".join(lines)

def run_freq(xyz_file, model_file, outfile=None, total_charge=0):
    "Calculate the energy and vibrational frequencies of a molecular geometry (either xyz or ase atoms) using a FENNIX model"
    import numpy as np
    
    atoms = _coerce_atoms(xyz_file)
    if isinstance(xyz_file, Atoms):
        print(f"Using ASE Atoms object with {len(atoms)} atoms for frequencies calculation.")
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

    # Center the coordinates at the center of mass
    atoms.set_positions(atoms.get_positions() - atoms.get_center_of_mass())

    # Align the principal axes of inertia with the coordinate axes by rotating into the principal-axes frame
    I, axes = atoms.get_moments_of_inertia(vectors=True)     # axes: rows = principal axes
    if np.linalg.det(axes) < 0:                              # avoid an improper rotation (mirror image)
        axes[2] *= -1
    atoms.set_positions(atoms.get_positions() @ axes.T)

    # Load the FENNIX model
    model_file = Path(model_file)
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=False)
    #convert = au.KCALPERMOL / model.Ha_to_model_energy 

    # Energy unit conversion: ase outputs in eV, convert to kcal/mol
    energy_conv = us.KCALPERMOL / us.EV
    atoms.calc = FENNIXCalculator(model, inputs=inputs, gpu_preprocessing=True)
    print(f"#DEBUG: Energy and max force before vibrational analysis: {atoms.get_potential_energy() * energy_conv:.6f} kcal/mol, max force = {np.abs(atoms.get_forces()).max() * energy_conv:.6f} kcal/mol/Å")

    vib = Vibrations(atoms)
    vib.run()
    report = _format_report(atoms, vib, energy_conv)

    vib.clean()
    print(report)
    if outfile is not None:
        outfile_path = Path(outfile)
        outfile_path.write_text(report)
    print("\nPhoBOOS vibrational frequency calculation completed successfully.")

    #Debug scan
    #import numpy as np
    #mode = vib.get_mode(0)                 # most imaginary mode
    #mode /= np.linalg.norm(mode)
    #X0 = atoms.get_positions().copy()
    #for s in np.linspace(-0.6, 0.6, 13):
    #    atoms.set_positions(X0 + s * mode)
    #    print(f"{s:+.2f}  {atoms.get_potential_energy():.5f}")
    #atoms.set_positions(X0)