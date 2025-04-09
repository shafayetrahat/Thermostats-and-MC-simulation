import matplotlib.pyplot as plt
from md_sim import run_simulation_py
import matplotlib.pyplot as plt

# Save 
if __name__ == "__main__":
    # Simulation parameters
    n_particles = 576
    dimensions = 2
    # density = 0.4
    # densities = [0.1,0.4,1]
    # T=0.1
    # Run simulation
    # temperature = [0.8,0.6,0.4,0.3,0.1]
    # for density in densities:
    density = 0.4
    steps = 3000

    # Run with trajectory saving


    # # Run with XYZ trajectory saving
    # results = run_simulation_py(
    #     steps=steps,
    #     trajectory_save=True,
    #     trajectory_file="my_traj.xyz",
    #     n_particles=n_particles,
    #     dimensions=dimensions,
    #     density=density,
    #     temperature=T,
    #     save_freq=5  # Save every 5 steps
    #     )

            # Plot temperature
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(results['temperature'])
        # plt.xlabel('Step')
        # plt.ylabel('Temperature')
        # plt.title('Temperature Evolution')
        # Plot RDF
        # plt.subplot(1, 2, 2)
    #     r, g_r = results['rdf']
    #     positions = results['positions']
        
    #     plt.plot(r, g_r,label=f'g(r) {density}')
    #     plt.xlabel('r')
    #     plt.ylabel('g(r)')
    #     plt.title('Radial Distribution Function (Last 2000 steps)')
    #     plt.tight_layout()
    #     plt.grid(True)
    #     plt.legend()
    # plt.savefig(f'test_densities/md_res_all.png')
    # Run with XYZ trajectory saving
    # for T in temperature:
    #     results = run_simulation_py(
    #     steps=steps,
    #     trajectory_save=False,
    #     n_particles=n_particles,
    #     dimensions=dimensions,
    #     density=density,
    #     temperature=T,
    #     compute_rdf_flag=True,
    #     )
    #     # Plot RDF
    #     r, g_r = results['rdf']
    #     # positions = results['positions']
    #     plt.plot(r, g_r,label=f'g(r) Temperature= {T}')
    #     plt.xlabel('r')
    #     plt.ylabel('g(r)')
    #     plt.title('Radial Distribution Function (Last 2000 steps)')
    #     plt.tight_layout()
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(f'test_temp/md_res_{T}.png')
    #     plt.clf()
    T=0.4
    results = run_simulation_py(
        steps=steps,
        trajectory_save=False,
        n_particles=n_particles,
        dimensions=dimensions,
        density=density,
        temperature=T,
        compute_rdf_flag=True,
        )