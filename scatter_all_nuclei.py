import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.linalg as la
from VQE.Nucleus import Nucleus
import pandas as pd

params = {'axes.linewidth': 1.4,
         'axes.labelsize': 16,
         'axes.titlesize': 18,
         'axes.linewidth': 1.5,
         'lines.markeredgecolor': "black",
     	'lines.linewidth': 1.5,
         'xtick.labelsize': 11,
         'ytick.labelsize': 14,
         "text.usetex": True,
         "font.family": "serif",
         "font.serif": ["Palatino"]
         }
plt.rcParams.update(params)

nuc_list_odd=['Li6','Li8','B8','Li10','N10','B10']
nuc_list_even=['Be6','He6','Be8','Be10','C10']

fig,ax = plt.subplots(1,2,figsize=(13,6))

for nuc_name in nuc_list_odd:
    nuc = Nucleus(nuc_name,1)
    d_H = nuc.d_H
    UCC_folder = (f'./outputs/{nuc.name}/v_performance/UCC_Reduced')
    ucc_file = os.listdir(UCC_folder)
    ucc_file = [f for f in ucc_file if f'L-BFGS-B' in f]
    ucc_data = pd.read_csv(os.path.join(UCC_folder,ucc_file[0]),sep='\t',header=None)
    ucc_data[[1,2,3,4]].astype(float)
    ucc_data = ucc_data[ucc_data[0]=='v0']
    ucc_depth = ucc_data[3]
    ucc_depth_std = ucc_data[4]
    ax[0].errorbar(d_H,ucc_depth,yerr=ucc_depth_std,fmt='o',label=f'{nuc.name}')

    adapt_folder = (f'./outputs/{nuc.name}/v_performance/ADAPT')
    adapt_file = os.listdir(adapt_folder)
    adapt_file = [f for f in adapt_file if 'basis.csv' in f]
    adapt_data = pd.read_csv(os.path.join(adapt_folder,adapt_file[0]),sep='\t')
    if 'Success' in adapt_data.columns:
        adapt_data=adapt_data[adapt_data['Success']=='SUCCESS']
    adapt_depth = adapt_data['Gates'].mean()
    adapt_depth_std = adapt_data['Gates'].std()

    ax[0].errorbar(d_H,adapt_depth,yerr=adapt_depth_std,marker='p',label=f'{nuc.name}')

for nuc_name in nuc_list_even:
    nuc = Nucleus(nuc_name,1)
    d_H = nuc.d_H
    UCC_folder = (f'./outputs/{nuc.name}/v_performance/UCC_Reduced')
    ucc_file = os.listdir(UCC_folder)
    ucc_file = [f for f in ucc_file if f'L-BFGS-B' in f]
    ucc_data = pd.read_csv(os.path.join(UCC_folder,ucc_file[0]),sep='\t',header=None)
    ucc_data[[1,2,3,4]].astype(float)
    ucc_data = ucc_data[ucc_data[0]=='v0']
    ucc_depth = ucc_data[3]
    ucc_depth_std = ucc_data[4]
    ax[1].errorbar(d_H,ucc_depth,yerr=ucc_depth_std,fmt='o',label=f'{nuc.name}')

    adapt_folder = (f'./outputs/{nuc.name}/v_performance/ADAPT')
    adapt_file = os.listdir(adapt_folder)
    adapt_file = [f for f in adapt_file if 'basis.csv' in f]
    adapt_data = pd.read_csv(os.path.join(adapt_folder,adapt_file[0]),sep='\t')
    if 'Success' in adapt_data.columns:
        adapt_data=adapt_data[adapt_data['Success']=='SUCCESS']
    adapt_depth = adapt_data['Gates'].mean()
    adapt_depth_std = adapt_data['Gates'].std()

    ax[1].errorbar(d_H,adapt_depth,yerr=adapt_depth_std,marker='p',label=f'{nuc.name}')

ax[0].legend()
ax[0].set_xlabel('Hilbert space dimension')
ax[1].set_xlabel('Hilbert space dimension')
ax[1].set_ylabel('Circuit depth')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
plt.show()
