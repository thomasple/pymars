"""
Example of using pymars with energy output tracking.
Similar to FeNNol's md.out energy files.
Uses new nested YAML-like configuration structure.
"""

import numpy as np
from pymars.md import initialize_collision_simulation

# Simulation parameters using nested structure
simulation_parameters = {
    "calculation_parameters": {
        "model": "path/to/fennix/model.pt",  # Update with actual model path
    },
    "input_parameters": {
        "initial_geometry": "tests/aspirin.xyz",
        "total_charge": 0,
        "random_rotation": True,
    },
    "general_parameters": {
        "temperature": 300.0,  # K
        "batch_size": 2,  # Run 2 parallel simulations
        "seed": 12345,
        "save_steps": 100,
        "save_energy": 100,
    },
    "projectile_parameters": {
        "projectile_flag": True,
        "projectile_species": 18,  # Argon
        "projectile_temperature": 3000.0,  # K
        "projectile_distance": 20.0,  # angstrom
        "max_impact_parameter": 0.5,  # angstrom
    },
    "thermostat_parameters": {
        "NVE_thermostat": True,
        "LGV_thermostat": False,
        "gamma": 0.0,  # THz
    },
    "dynamic_parameters": {
        "dt_dyn": 1.0,  # fs
        "step_dyn": 10000,
    },
}

# Initialize simulation
state = initialize_collision_simulation(simulation_parameters, verbose=True)

# Extract simulation state
coordinates = state["coordinates"]
velocities = state["velocities"]
accelerations = state["accelerations"]
integrate = state["integrate"]
initial_energies = state["initial_energies"]

# Print initial total energy
print("\n# Initial energies:")
for b, E in enumerate(initial_energies):
    print(f"  Trajectory {b}: E_tot = {float(E):.6f} kcal/mol")

# Setup energy output files
energy_output_files = ["energy_traj_0.out", "energy_traj_1.out"]
save_energy = 100  # Write every 100 steps

# Run MD simulation
n_steps = 10000
for step in range(n_steps):
    coordinates, velocities, accelerations, energies = integrate(
        coordinates, 
        velocities, 
        accelerations,
        step=step,
        energy_output_file=energy_output_files,
        energy_steps=save_energy
    )
    
    if step % 1000 == 0:
        print(f"Step {step}/{n_steps}")

print("\nSimulation complete!")
print(f"Energy data written to: {', '.join(energy_output_files)}")
