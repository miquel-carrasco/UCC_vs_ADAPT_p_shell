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

nuc_list=['Li6','Li8','B8','Li10','N10','B10','Be6','He6','Be8','Be10','C10']
x=[9,27.5,28.5,10,11,81,4.5,5.5,50,51,52]
colors=['tab:green','tab:red','tab:red','tab:green','tab:green','tab:brown','tab:blue','tab:blue','tab:orange','tab:orange','tab:orange']
labels=[r'$d_{\hat{H}}=10$','28',None,None,None,'81','5',None,'51',None,None]

fig,ax = plt.subplots(1,1,figsize=(8,6))

for i,nuc_name in enumerate(nuc_list):
    nuc = Nucleus(nuc_name,1)
    d_H = nuc.d_H
    UCC_folder = (f'./outputs/{nuc.name}/v_performance/UCC_Reduced')
    ucc_file = os.listdir(UCC_folder)
    ucc_file = [f for f in ucc_file if f'L-BFGS-B' in f]
    ucc_data = pd.read_csv(os.path.join(UCC_folder,ucc_file[0]),sep='\t',header=None)
    ucc_data[[1,2,3,4]].astype(float)
    ucc_data = ucc_data[ucc_data[0]=='random']
    ucc_depth = ucc_data[3]
    ucc_depth_std = ucc_data[4]
    ax.errorbar(x[i],ucc_depth,yerr=ucc_depth_std,fmt='o',label=labels[i],color=colors[i])

    adapt_folder = (f'./outputs/{nuc.name}/v_performance/ADAPT')
    adapt_file = os.listdir(adapt_folder)
    adapt_file = [f for f in adapt_file if 'basis.csv' in f]
    adapt_data = pd.read_csv(os.path.join(adapt_folder,adapt_file[0]),sep='\t')
    if 'Success' in adapt_data.columns:
        adapt_data=adapt_data[adapt_data['Success']=='SUCCESS']
    adapt_depth = adapt_data['Gates'].mean()
    adapt_depth_std = adapt_data['Gates'].std()

    ax.errorbar(x[i],adapt_depth,yerr=adapt_depth_std,marker='p',label=labels[i],color=colors[i])


ax.legend()
ax.set_xlabel('Hilbert space dimension')
ax.set_ylabel('Circuit depth')
ax.set_yscale('log')
plt.show()
