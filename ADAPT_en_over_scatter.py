import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy.linalg as la
from VQE.Nucleus import Nucleus

params = {'axes.linewidth': 1.4,
         'axes.labelsize': 16,
         'axes.titlesize': 18,
         'axes.linewidth': 1.5,
         'lines.markeredgecolor': "black",
     	'lines.linewidth': 1.5,
         'xtick.labelsize': 12,
         'ytick.labelsize': 12,
         "text.usetex": True,
         "font.family": "serif",
         "font.serif": ["Palatino"]
         }
plt.rcParams.update(params)

nucleus = Nucleus('B10',1)
d_H = nucleus.d_H

files_folder = f'./outputs/{nucleus.name}/v_performance/ADAPT'
files = os.listdir(files_folder)

basis_file = [f for f in files if 'basis.csv' in f][0]
random_file = [f for f in files if 'random.csv' in f][0]

basis_df = pd.read_csv(f'{files_folder}/{basis_file}', sep='\t')
random_df = pd.read_csv(f'{files_folder}/{random_file}', sep='\t')

E_basis = np.array(basis_df['E0'])
overlap_basis = np.array(basis_df['Overlap'])
gates_basis = np.array(basis_df['Gates'])


E_random = random_df['E0']
overlap_random = random_df['Overlap']
gates_random = random_df['Gates']

### FIGURE 1 ###
fig, ax = plt.subplots(1,2, figsize=(13,6), sharey=True)
ax[0].scatter(E_basis, gates_basis, marker = 'p', color = 'tab:blue', label='Basis states')
ax[0].scatter(E_random, gates_random, marker = 's', color = 'tab:red', label='Random states')
ax[0].set_xlabel('Ref. state energy')
ax[0].set_ylabel('Circuit depth')


ax[1].scatter(overlap_basis, gates_basis, marker = 'p', color = 'tab:blue', label='Basis states')
ax[1].scatter(overlap_random, gates_random, marker = 's', color = 'tab:red', label='Random states')
ax[1].set_xlabel('Ref. state overlap')
ax[1].set_xscale('log')
ax[1].legend()
fig.subplots_adjust(wspace=0.05)
fig.suptitle(f'Reference state performance according to energy and overlap, {nucleus.name}',fontsize=18)
fig.savefig(f'./figures/{nucleus.name}/ADAPT_energy_overlap.pdf', bbox_inches='tight')

plt.close()



### FIGURE 2 ###
plt.scatter(E_basis, gates_basis, c=overlap_basis, cmap='viridis', marker='p', label='Basis states')
plt.scatter(E_random, gates_random, c=overlap_random, cmap='viridis', marker='s', label='Random states')
cbar = plt.colorbar()
cbar.set_label('Ref. state overlap')
plt.xlabel('Ref. state energy')
plt.ylabel('Circuit depth')
plt.title(f'Reference state performance according to energy and overlap, {nucleus.name}')
plt.savefig(f'./figures/{nucleus.name}/ADAPT_energy_overlap_heatmap.pdf', bbox_inches='tight')
plt.close()
