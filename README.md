# Thermostats-and-MC-simulation
This repository is about applying a thermostat to MD simulations and comparing MD simulation with MC simulation


# Directory Structure
```
.
├── LICENSE
├── problem1
│   └── Part1_2.ipynb
├── problem2
│   └── Part2.ipynb
├── problem3
│   ├── cython
│   └── python
├── problem4
│   ├── cython
│   └── python
├── README.md
└── utils
    └── barplot.py
```
# How to run
### For .ipynb files,  
Go to the <base_directory/problemdir/ > directory and run it with jupyter-lab

### For Python files,
Go to the <base_directory/problemdir/python/ > and run   
python3 <filename.py>

### For Cython files,
To compile Cython file,
Go to the <base_directory/problemdir/cython/ > and run 
- python setup.py build_ext --inplace
To run the simulation, go to the <base_directory/problemdir/cython/ > and run, 
- python3 main.py
  
# Configuration and Control variable description
For Problem 1
```
Described inside the .ipynb file
```
For Problem 2
```
Described in the .ipynb file
```
For Problem 3
```
n_particles
    - total number of LJ particles
dimensions
    - Dimension of the box
density
    - Density of the box. N/L^2

T
    - Temperature

steps
    - Total steps of the simulation
dt
    - Timestep

Other optional flags are described in the Appendix section of the report. 

```
For Problem 4
```
N
    - Number of particles
L
    - Box length
T
    - Temperature
rc
    - Cutoff distance
max_disp
    - Maximum displacement radius
steps
    - Total MC steps
equil_steps
    - Equilibration steps
sample_freq
    - Sampling frequency
```
# Simulation trajectory from MD simulation
The trajectory video from the MD simulation is given below.


https://github.com/user-attachments/assets/c5c2b37b-8047-4e61-b2f9-12d3beaf5d20

