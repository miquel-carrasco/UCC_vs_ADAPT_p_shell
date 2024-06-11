import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy.linalg as la
from VQE.Nucleus import Nucleus

# params = {'axes.linewidth': 1.4,
#          'axes.labelsize': 16,
#          'axes.titlesize': 18,
#          'axes.linewidth': 1.5,
#          'lines.markeredgecolor': "black",
#      	'lines.linewidth': 1.5,
#          'xtick.labelsize': 11,
#          'ytick.labelsize': 14,
#          "text.usetex": True,
#          "font.family": "serif",
#          "font.serif": ["Palatino"]
#          }
# plt.rcParams.update(params)

nucleus = Nucleus('Be8',1)
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
plt.scatter(E_random, gates_random, c=overlap_random, cmap='viridis', marker='o', label='Basis states')
plt.savefig(f'prova.pdf')
