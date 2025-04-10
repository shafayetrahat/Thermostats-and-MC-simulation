# Thermostats-and-MC-simulation
This repository is about applying a thermostat to MD simulations and comparing MD simulation with MC simulation

# Dependencies
```
Cython==3.0.12
matplotlib==3.7.2
matplotlib-inline==0.1.6
ipykernel==6.21.1
ipython==8.10.0
jupyter==1.0.0
jupyter-console==6.5.0
jupyter-events==0.5.0
jupyter-ydoc==0.2.2
jupyter_client==8.0.2
jupyter_core==5.2.0
jupyter_server==2.2.1
jupyter_server_fileid==0.6.0
jupyter_server_terminals==0.4.4
jupyter_server_ydoc==0.6.1
jupyterlab==3.6.1
jupyterlab-pygments==0.2.2
jupyterlab-widgets==3.0.5
jupyterlab_server==2.19.0
```
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

