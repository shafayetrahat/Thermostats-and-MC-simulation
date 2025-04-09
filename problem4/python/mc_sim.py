import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(N, L, rho):
    """Initialize particles on triangular lattice with random fill"""
    a = np.sqrt(2/(rho*np.sqrt(3)))  # Lattice spacing
    nx, ny = int(L/a), int(L/(a*np.sqrt(3)/2))
    pos = []
    
    # Create lattice positions
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5*(j%2)) * a
            y = j * a * np.sqrt(3)/2
            if x < L and y < L and len(pos) < N:
                pos.append([x, y])
    
    # Fill remaining randomly
    while len(pos) < N:
        pos.append(np.random.rand(2) * L)
    
    positions = np.array(pos)
    return positions, calculate_total_energy(positions, N, L, rc)

def periodic_distance(r1, r2, L):
    """Minimum image convention distance"""
    dr = r1 - r2
    dr -= L * np.round(dr / L)
    return np.sqrt(np.sum(dr**2))

def lj_potential(r, rc):
    """Lennard-Jones potential with cutoff"""
    return 4 * ((1/r)**12 - (1/r)**6) if r < rc else 0

def calculate_total_energy(positions, N, L, rc):
    """Total potential energy of the system"""
    e = 0
    for i in range(N):
        for j in range(i+1, N):
            r = periodic_distance(positions[i], positions[j], L)
            e += lj_potential(r, rc)
    return e

def mc_move(positions, N, L, rc, max_disp, beta):
    """MC move with circular displacement using polar coordinates"""
    # Select random particle
    particle = np.random.randint(N)
    old_pos = positions[particle].copy()
    
    # Generate random displacement within circle
    theta = 2 * np.pi * np.random.rand()      # Random angle [0, 2π]
    r = max_disp * np.sqrt(np.random.rand())  # Uniform in area
    dx, dy = r * np.cos(theta), r * np.sin(theta)
    displacement = np.array([dx, dy])
    
    # New position with PBC
    new_pos = (old_pos + displacement) % L
    
    # Calculate energy change
    deltaE = 0
    for j in range(N):
        if j != particle:
            r_old = periodic_distance(old_pos, positions[j], L)
            r_new = periodic_distance(new_pos, positions[j], L)
            deltaE += lj_potential(r_new, rc) - lj_potential(r_old, rc)
    
    # Metropolis criterion
    max_exp = 40
    if deltaE < 0 and np.random.rand() < np.exp(min((-beta * deltaE), max_exp)):
        # Accept move
        positions[particle] = new_pos
        return positions, deltaE, True
    else:
        return positions, 0, False

def run_simulation(N, L, T, rc, max_disp, steps, equil_steps, sample_freq):
    """Main simulation routine"""
    rho = N/L**2
    beta = 1.0/T
    
    # Initialize
    print("Initializing system...")
    positions, energy = initialize_lattice(N, L, rho)
    
    # Equilibration
    print(f"Burn-in steps {equil_steps} ...")
    for step in range(equil_steps):
        positions, deltaE, accepted = mc_move(positions, N, L, rc, max_disp, beta)
        energy += deltaE
        if step % 500 == 0:
            print(f"Step {step}/{equil_steps}")
    
    # Production run
    print(f"Production run for {steps} steps...")
    energies = []
    acc_count = 0
    for step in range(steps):
        positions, deltaE, accepted = mc_move(positions, N, L, rc, max_disp, beta)
        if accepted==False:
            continue
        energy += deltaE
        if accepted:
            acc_count += 1 
        if step % sample_freq == 0 and accepted:
            energies.append(energy)
        if step % 2000 == 0:
            print(f"Completed {step}/{steps} steps")
    
    print(f"Acceptance rate: {acc_count/steps:.3f}")
    return positions, np.array(energies)

# Save positions to .xyz file
def save_xyz(positions, filename="mc_final.xyz"):
    with open(filename, "a") as f:
        f.write(f"{len(positions)}\n")
        f.write(f"LJ-MC final conformation\n")
        for i, pos in enumerate(positions, start=1):
            f.write(f"X {pos[0]} {pos[1]} 0.0\n")   
    
def plot_results(positions, energies):
    """Plot simulation results"""
    plt.figure(figsize=(15, 5))
    
    # Energy trajectory
    plt.subplot(1, 3, 1)
    plt.plot(energies)
    plt.xlabel("MC step (sampled)")
    plt.ylabel("Energy")
    plt.title("Energy Trajectory")
    
    # Final configuration
    plt.subplot(1, 3, 2)
    plt.scatter(positions[:,0], positions[:,1], s=30)
    plt.title("Final Configuration")
    plt.xlabel("x"); plt.ylabel("y")
    
    # Radial distribution
    plt.subplot(1, 3, 3)
    distances = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            distances.append(periodic_distance(positions[i], positions[j], L))
    plt.hist(distances, bins=30, density=True)
    plt.title("Pair Distance Distribution")
    plt.xlabel("Distance")
    
    plt.tight_layout()
    plt.savefig("mc_simulation_results.png")

if __name__ == "__main__":
    # Simulation parameters
    N = 576                  # Number of particles
    L = 24.0                 # Box length
    T = 0.1                # Temperature
    rc = 2.5                 # Cutoff distance
    max_disp = 0.99          # Maximum displacement radius
    steps = 3000             # Total MC steps
    equil_steps = 500       # Equilibration steps
    sample_freq = 100        # Sampling frequency
    
    # Run simulation
    positions, energies = run_simulation(N, L, T, rc, max_disp, steps, equil_steps, sample_freq
    )
    
    # Plot results
    plot_results(positions, energies)
    # Save final positions to .xyz file
    save_xyz(positions)
    # Print final energy
    print(f"Final energy: {energies[-1]:.2f}")