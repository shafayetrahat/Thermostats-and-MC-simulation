# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
import cython
from libc.math cimport sqrt, pow, cos, sin, exp, round as cround
from libc.stdlib cimport rand, RAND_MAX

# Type definitions
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

# Initialize random seed
cdef extern from "stdlib.h":
    void srand(unsigned int seed)

srand(42)

@cython.boundscheck(False)
@cython.wraparound(False)
def initialize_lattice(int N, DTYPE_t L, DTYPE_t rho, DTYPE_t rc):
    """Initialize particles on triangular lattice with random fill"""
    cdef DTYPE_t a = sqrt(2/(rho*sqrt(3)))  # Lattice spacing
    cdef int nx = int(L/a), ny = int(L/(a*sqrt(3)/2))
    cdef list pos = []
    cdef int i, j
    cdef DTYPE_t x, y
    
    # Create lattice positions
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5*(j%2)) * a
            y = j * a * sqrt(3)/2
            if x < L and y < L and len(pos) < N:
                pos.append([x, y])
    
    # Fill remaining randomly - updated array access
    cdef np.ndarray[DTYPE_t, ndim=2] random_pos = np.random.rand(N - len(pos), 2) * L
    cdef int random_pos_size = random_pos.shape[0]  # Use shape attribute instead of direct struct access
    
    for i in range(random_pos_size):
        pos.append([random_pos[i, 0], random_pos[i, 1]])
    
    cdef np.ndarray[DTYPE_t, ndim=2] positions = np.array(pos, dtype=DTYPE)
    return positions, calculate_total_energy(positions, N, L, rc)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_t periodic_distance(np.ndarray[DTYPE_t, ndim=1] r1, 
                              np.ndarray[DTYPE_t, ndim=1] r2, 
                              DTYPE_t L):
    """Minimum image convention distance"""
    cdef np.ndarray[DTYPE_t, ndim=1] dr = r1 - r2
    dr -= L * cround(dr[0] / L)
    dr -= L * cround(dr[1] / L)
    return sqrt(dr[0]*dr[0] + dr[1]*dr[1])

@cython.cdivision(True)
cdef DTYPE_t lj_potential(DTYPE_t r, DTYPE_t rc):
    """Lennard-Jones potential with cutoff"""
    if r >= rc:
        return 0.0
    cdef DTYPE_t r6 = pow(r, -6)
    cdef DTYPE_t r12 = r6 * r6
    return 4.0 * (r12 - r6)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t calculate_total_energy(np.ndarray[DTYPE_t, ndim=2] positions, 
                                   int N, DTYPE_t L, DTYPE_t rc):
    """Total potential energy of the system"""
    cdef DTYPE_t e = 0.0
    cdef int i, j
    cdef DTYPE_t r
    
    for i in range(N):
        for j in range(i+1, N):
            r = periodic_distance(positions[i], positions[j], L)
            e += lj_potential(r, rc)
    return e

@cython.boundscheck(False)
@cython.wraparound(False)
def mc_move(np.ndarray[DTYPE_t, ndim=2] positions, 
            int N, DTYPE_t L, DTYPE_t rc, 
            DTYPE_t max_disp, DTYPE_t beta):
    """MC move with circular displacement using polar coordinates"""
    # Select random particle
    cdef int particle = rand() % N
    cdef np.ndarray[DTYPE_t, ndim=1] old_pos = positions[particle].copy()
    
    # Generate random displacement within circle
    cdef DTYPE_t theta = 2 * np.pi * (<DTYPE_t>rand() / RAND_MAX)
    cdef DTYPE_t r = max_disp * sqrt(<DTYPE_t>rand() / RAND_MAX)
    cdef DTYPE_t dx = r * cos(theta)
    cdef DTYPE_t dy = r * sin(theta)
    cdef np.ndarray[DTYPE_t, ndim=1] displacement = np.array([dx, dy], dtype=DTYPE)
    
    # New position with PBC
    cdef np.ndarray[DTYPE_t, ndim=1] new_pos = (old_pos + displacement) % L
    
    # Calculate energy change
    cdef DTYPE_t deltaE = 0.0
    cdef int j
    cdef DTYPE_t r_old, r_new
    
    for j in range(N):
        if j != particle:
            r_old = periodic_distance(old_pos, positions[j], L)
            r_new = periodic_distance(new_pos, positions[j], L)
            deltaE += lj_potential(r_new, rc) - lj_potential(r_old, rc)
    
    # Metropolis criterion
    cdef DTYPE_t max_exp = 40.0
    
    if deltaE < 0 and (<DTYPE_t>rand() / RAND_MAX) < exp(min(-beta * deltaE, max_exp)):
        # Accept move
        positions[particle, 0] = new_pos[0]
        positions[particle, 1] = new_pos[1]
        return positions, deltaE, True
    else:
        return positions, 0.0, False

def run_simulation(int N, DTYPE_t L, DTYPE_t T, DTYPE_t rc, 
                   DTYPE_t max_disp, int steps, int equil_steps, int sample_freq):
    """Main simulation routine"""
    cdef DTYPE_t rho = N/(L*L)
    cdef DTYPE_t beta = 1.0/T
    
    # Initialize
    print("Initializing system...")
    cdef np.ndarray[DTYPE_t, ndim=2] positions
    cdef DTYPE_t energy
    positions, energy = initialize_lattice(N, L, rho, rc)
    
    # Equilibration
    print(f"Burn-in steps {equil_steps} ...")
    cdef int step
    cdef DTYPE_t deltaE
    cdef bint accepted
    
    for step in range(equil_steps):
        positions, deltaE, accepted = mc_move(positions, N, L, rc, max_disp, beta)
        energy += deltaE
        if step % 500 == 0:
            print(f"Step {step}/{equil_steps}")
    
    # Production run
    print(f"Production run for {steps} steps...")
    cdef list energies = []
    cdef int acc_count = 0
    
    for step in range(steps):
        positions, deltaE, accepted = mc_move(positions, N, L, rc, max_disp, beta)
        if accepted==False:
            continue
        energy += deltaE
        if accepted:
            acc_count += 1 
        if step % sample_freq == 0 and accepted:
            energies.append(energy)
    
    print(f"Completed {steps} steps")
    print(f"Acceptance rate: {acc_count/steps:.3f}")
    return positions, np.array(energies, dtype=DTYPE), acc_count/steps