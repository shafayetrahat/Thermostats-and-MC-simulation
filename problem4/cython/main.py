from mc_sim import run_simulation, periodic_distance
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd

# Save positions to .xyz file
def save_xyz(positions, filename="mc_final.xyz"):
    with open(filename, "a") as f:
        f.write(f"{len(positions)}\n")
        f.write(f"LJ-MC final conformation\n")
        for i, pos in enumerate(positions, start=1):
            f.write(f"X {pos[0]} {pos[1]} 0.0\n")   

def compute_gr(positions, box_size, rcutoff=5.0, n_bins=100):
    # Combine all samples (if multiple frames were provided)
    if len(positions.shape) == 3:  # Multiple samples
        positions = positions.reshape(-1, 2)
    
    n_particles = positions.shape[0]
    dr = rcutoff / n_bins
    gr_hist = np.zeros(n_bins)
    
    # Compute all pairwise distances
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r = periodic_distance(positions[i], positions[j], box_size)
            if r < rcutoff:
                bin_index = int(r / dr)
                gr_hist[bin_index] += 2  # Count each pair only once
    
    # Normalization for 2D system
    r_values = dr * (np.arange(n_bins) + 0.5)  # bin centers
    shell_areas = 2 * np.pi * r_values * dr  # area of each annular shell
    density = n_particles / (box_size ** 2)  # number density
    ideal_counts = shell_areas * density * n_particles  # expected counts for ideal gas
    
    # Avoid division by zero for r=0
    nonzero = ideal_counts > 0
    gr_values = np.zeros_like(r_values)
    gr_values[nonzero] = gr_hist[nonzero] / ideal_counts[nonzero]
    
    return r_values, gr_values


# if __name__ == "__main__":
#     # Simulation parameters
#     N = 576                 # Number of particles
#     L = 50                 # Box length
#     T = 0.1                # Temperature
#     rc = 2.5                 # Cutoff distance
#     max_disp = 0.05          # Maximum displacement radius
#     steps = 30000             # Total MC steps
#     equil_steps = 1000       # Equilibration steps
#     sample_freq = 100        # Sampling frequency
    
#     # Run simulation
#     positions, energies,acceptence_rate = run_simulation(N, L, T, rc, max_disp, steps, equil_steps, sample_freq)

#     # Save final positions to .xyz file
#     save_xyz(positions)

# # Take last 100 samples to compute g(r)
#     print("Computing radial distribution function g(r)...")
#     last_samples = positions[-500:] if len(positions) >= 500 else positions
#     r_vals, gr_vals = compute_gr(last_samples, L)

#     # Plot g(r)
#     plt.figure(figsize=(6, 4))
#     plt.plot(r_vals, gr_vals, label='g(r)')
#     plt.xlabel('r')
#     plt.ylabel('g(r)')
#     plt.title('Radial Distribution Function (Last 100 Samples)')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# Ls=[24,28,32,40,150]

# for L in Ls:
#     # Run simulation
#     positions, energies,acceptence_rate = run_simulation(N, L, T, rc, max_disp, steps, equil_steps, sample_freq)
#     print(f'Final energy: {energies[-1]}')
#     # Save final positions to .xyz file
#     # save_xyz(positions)
#     #plot g(r) and final positions
#     last_samples = positions[-2000:] if len(positions) >= 2000 else positions
#     r_vals, gr_vals = compute_gr(last_samples, L)
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.plot(energies)
#     plt.xlabel("MC step (samples)x1000")
#     plt.ylabel("Energy")
#     plt.title("Energy Trajectory")
#     plt.tight_layout()
#     plt.grid(True)
#     plt.subplot(1, 3, 2)
#     plt.plot(r_vals, gr_vals, label='g(r)')
#     plt.xlabel('r')
#     plt.ylabel('g(r)')
#     plt.title(f'Radial Distribution Function (L={L})')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.subplot(1, 3, 3)
#     plt.scatter(positions[:,0], positions[:,1], s=30)
#     plt.title(f'Final Configuration (L={L})')
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.tight_layout()
#     plt.savefig(f"density_test/mc_res_L{L}.png")
# Simulation parameters
N = 576                 # Number of particles
L =40                 # Box length
# T = 0.1                # Temperature
rc = 2.5                 # Cutoff distance
max_disp = 0.95          # Maximum displacement radius
steps = 3000             # Total MC steps
equil_steps = 100       # Equilibration steps
sample_freq = 100        # Sampling frequency
    
Ts=[0.1] # Temperature
for T in Ts:
    # Run simulation
    positions, energies,acceptence_rate = run_simulation(N, L, T, rc, max_disp, steps, equil_steps, sample_freq)
    print(f'Final energy: {energies[-1]}')
    # Save final positions to .xyz file
    # save_xyz(positions)
    #plot g(r) and final positions
    last_samples = positions[-100:] if len(positions) >= 100 else positions
    r_vals, gr_vals = compute_gr(last_samples, L)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(energies)
    plt.xlabel("MC step (samples)")
    plt.ylabel("Energy")
    plt.title("Energy Trajectory")
    plt.tight_layout()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(r_vals, gr_vals, label='g(r)')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title(f'Radial Distribution Function (L={L})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    plt.scatter(positions[:,0], positions[:,1], s=30)
    plt.title(f'Final Configuration (L={L})')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"temp_test/mc_res_t{T}.png")