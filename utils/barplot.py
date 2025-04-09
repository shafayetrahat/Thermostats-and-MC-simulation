import matplotlib.pyplot as plt

fig, ax = plt.subplots()

simtype = ['Python', 'Cython']
#MC vs MD python
# counts = [ 694.98,71.67]
#MD python vs cython
# counts = [ 694.98, 15.684]
counts = [ 71.67,  20.28]
bar_labels = ['Python', 'Cython']
bar_colors = ['tab:red', 'tab:blue']

ax.bar(simtype, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('Time for 3000 cycle')
# ax.set_title(f'MC vs MD simulation time. MC is {counts[0]/counts[1]:.2f} times faster')

ax.set_title(f'Python vs Cython simulation time(MC). Cython is {counts[0]/counts[1]:.2f} times faster')
ax.legend(title='type')

plt.show()
