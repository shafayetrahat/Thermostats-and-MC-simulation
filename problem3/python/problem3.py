import numpy as np
import matplotlib.pyplot as plt


def compute_lj_force(r, sigma, epsilon, rcutoff):
    if r >= rcutoff or r < 1e-12:
        return 0.0
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    return 24 * epsilon * (2 * inv_r12 - inv_r6) / r


def compute_lj_potential(r, sigma, epsilon, rcutoff):
    if r >= rcutoff:
        return 0.0
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    pot = 4 * epsilon * (inv_r12 - inv_r6)
    inv_rcut = sigma / rcutoff
    shift = 4 * epsilon * (inv_rcut ** 12 - inv_rcut ** 6)
    return pot - shift


def build_linked_cells(positions, box_size, rcutoff):
    n_particles, dim = positions.shape
    lc = max(1, int(np.floor(box_size / rcutoff)))
    lc_dim = [lc] * dim
    rc = box_size / lc
    EMPTY = -1
    head = [EMPTY] * (lc ** dim)
    lscl = [EMPTY] * n_particles

    for i in range(n_particles):
        mc = [int(positions[i][a] / rc) for a in range(dim)]
        mc = [min(max(0, idx), lc - 1) for idx in mc]
        if dim == 2:
            c_index = mc[0] * lc_dim[1] + mc[1]
        else:
            c_index = mc[0] * lc_dim[1] * lc_dim[2] + mc[1] * lc_dim[2] + mc[2]
        lscl[i] = head[c_index]
        head[c_index] = i

    return head, lscl, lc_dim


def compute_forces_lca(positions, box_size, rcutoff, sigma, epsilon, use_pbc=True):
    n_particles, dim = positions.shape
    head, lscl, lc_dim = build_linked_cells(positions, box_size, rcutoff)
    EMPTY = -1
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    neighbor_offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * dim)).T.reshape(-1, dim)

    for mc in np.ndindex(*lc_dim):
        if dim == 2:
            c_index = mc[0] * lc_dim[1] + mc[1]
        else:
            c_index = mc[0] * lc_dim[1] * lc_dim[2] + mc[1] * lc_dim[2] + mc[2]
        i = head[c_index]
        while i != EMPTY:
            pos_i = positions[i]
            for offset in neighbor_offsets:
                mc1 = np.array(mc) + offset
                rshift = np.zeros(dim)
                valid_cell = True
                for a in range(dim):
                    if use_pbc:
                        if mc1[a] < 0:
                            mc1[a] += lc_dim[a]
                            rshift[a] = -box_size
                        elif mc1[a] >= lc_dim[a]:
                            mc1[a] -= lc_dim[a]
                            rshift[a] = box_size
                    else:
                        if mc1[a] < 0 or mc1[a] >= lc_dim[a]:
                            valid_cell = False
                            break
                if not valid_cell:
                    continue
                if dim == 2:
                    c1 = mc1[0] * lc_dim[1] + mc1[1]
                else:
                    c1 = mc1[0] * lc_dim[1] * lc_dim[2] + mc1[1] * lc_dim[2] + mc1[2]
                j = head[c1]
                while j != EMPTY:
                    if j > i:
                        pos_j = positions[j] + rshift
                        r_ij = pos_i - pos_j
                        dist = np.linalg.norm(r_ij)
                        if dist < rcutoff and dist > 1e-12:
                            f_mag = compute_lj_force(dist, sigma, epsilon, rcutoff)
                            fij = f_mag * (r_ij / dist)
                            forces[i] += fij
                            forces[j] -= fij
                            potential_energy += compute_lj_potential(dist, sigma, epsilon, rcutoff)
                    j = lscl[j]
            i = lscl[i]

    return forces, potential_energy


def create_lattice(n_particles, box_size, dimensions):
    n_side = int(np.ceil(n_particles ** (1 / dimensions)))
    spacing = box_size / n_side
    positions = []
    for indices in np.ndindex(*([n_side] * dimensions)):
        if len(positions) < n_particles:
            pos = [(i + 0.5) * spacing for i in indices]
            noise = np.random.uniform(-0.05, 0.05, size=dimensions) * spacing
            positions.append(np.array(pos) + noise)
    return np.array(positions)


def initialize_velocities(n_particles, dimensions, target_temp):
    velocities = np.random.normal(0, 1, size=(n_particles, dimensions))
    velocities -= np.mean(velocities, axis=0)
    ke = 0.5 * np.sum(velocities**2)
    dof = n_particles * dimensions
    scale = np.sqrt(target_temp * dof / ke)
    velocities *= scale
    return velocities


def apply_berendsen_thermostat(velocities, target_temp, current_temp, dt, tau):
    lambda_factor = np.sqrt(1 + dt / tau * (target_temp / current_temp - 1))
    return velocities * lambda_factor


def apply_langevin_thermostat(velocities, gamma, dt, target_temp):
    noise = np.random.normal(0, 1, size=velocities.shape)
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1**2) * target_temp)
    return c1 * velocities + c2 * noise


def compute_rdf(positions, box_size, rcutoff, n_bins):
    n_particles = positions.shape[0]
    dr = rcutoff / n_bins
    rdf_hist = np.zeros(n_bins)
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rij = positions[i] - positions[j]
            rij -= box_size * np.round(rij / box_size)  # PBC
            r = np.linalg.norm(rij)
            if r < rcutoff:
                bin_index = int(r / dr)
                rdf_hist[bin_index] += 2

    density = n_particles / box_size**2
    r_values = dr * (np.arange(n_bins) + 0.5)
    shell_areas = 2 * np.pi * r_values * dr
    ideal_counts = density * shell_areas * n_particles
    g_r = rdf_hist / ideal_counts
    return r_values, g_r


def run_simulation(params, compute_rdf_flag=False, rdf_sample_steps=200, n_bins=100):
    box_size = (params['n_particles'] / params['density']) ** (1 / params['dimensions'])
    positions = create_lattice(params['n_particles'], box_size, params['dimensions'])
    velocities = initialize_velocities(params['n_particles'], params['dimensions'], params['temperature'])
    dof = params['n_particles'] * params['dimensions']

    temp_list = []
    rdf_accum = np.zeros(n_bins)
    rdf_count = 0

    for step in range(params['steps']):
        forces, _ = compute_forces_lca(positions, box_size, params['rcutoff'],
                                       params['sigma'], params['epsilon'], use_pbc=True)
        velocities += 0.5 * forces * params['dt']
        positions += velocities * params['dt']
        positions %= box_size
        forces, _ = compute_forces_lca(positions, box_size, params['rcutoff'],
                                       params['sigma'], params['epsilon'], use_pbc=True)
        velocities += 0.5 * forces * params['dt']

        kinetic = 0.5 * np.sum(velocities ** 2)
        temp = 2 * kinetic / dof
        temp_list.append(temp)

        if params['thermostat_type'] == 'berendsen':
            velocities = apply_berendsen_thermostat(velocities, params['temperature'], temp, params['dt'], params['tau_ber'])
        elif params['thermostat_type'] == 'langevin':
            velocities = apply_langevin_thermostat(velocities, params['gamma_langevin'], params['dt'], params['temperature'])

        if compute_rdf_flag and step >= params['steps'] - rdf_sample_steps:
            r_vals, gr = compute_rdf(positions, box_size, params['rcutoff'], n_bins)
            rdf_accum += gr
            rdf_count += 1

    rdf_avg = rdf_accum / rdf_count if rdf_count > 0 else None
    return temp_list, r_vals, rdf_avg


# ---- Multi-Temperature RDF Sweep ----
temperatures = [0.1]
density = 0.1

plt.figure(figsize=(8, 5))

for T in temperatures:
    params = {
        'dimensions': 2,
        'n_particles': 576,
        'density': density,
        'dt': 0.002,
        'steps': 3000,
        'temperature': T,
        'sigma': 1.0,
        'epsilon': 1.0,
        'rcutoff': 2.5,
        'tau_ber': 0.1,
        'gamma_langevin': 1.0,
        'thermostat_type': 'langevin'
    }
    _, r_vals, g_r = run_simulation(params, compute_rdf_flag=True)
    if g_r is not None:
        plt.plot(r_vals, g_r, label=f"T = {T}")

    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title(f'Radial Distribution Function at ρ = {density}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"temp_test/rdf_temp_{T}.png")
    plt.clf()
