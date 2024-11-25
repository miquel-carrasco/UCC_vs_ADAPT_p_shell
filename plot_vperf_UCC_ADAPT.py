import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.linalg as la
from VQE.Nucleus import Nucleus
import pandas as pd

params = {'axes.linewidth': 1.4,
         'axes.labelsize': 15,
         'axes.titlesize': 16,
         'axes.linewidth': 1.5,
         'lines.markeredgecolor': "black",
     	'lines.linewidth': 1.5,
         'xtick.labelsize': 11,
         'ytick.labelsize': 11,
         "text.usetex": True,
         "font.family": "serif",
         "font.serif": ["Palatino"]
         }
plt.rcParams.update(params)

nucleus = Nucleus('B8',1)
d_H = nucleus.d_H

runs=50


colors=['tab:blue','tab:orange','tab:green','tab:red']

fig, ax1 = plt.subplots(figsize=(11,6))
ax2=ax1.twinx()

x = [f'$v_{{{i}}}$' for i in np.arange(d_H)]
x.append('Rand')

infidelity=[]
for i in range(d_H):
    vi = np.eye(d_H)[:,i]
    overlapp=np.abs(nucleus.eig_vec[:,0].conj().T@vi)**2
    infidelity.append(1-overlapp)
ax1.bar(x[:-1],infidelity,alpha=0.5,color='grey',label='Infidelity',zorder=1,edgecolor='black')


UCC_folder = (f'./outputs/{nucleus.name}/v_performance/UCC_ReducedII')
ucc_file = os.listdir(UCC_folder)
ucc_file = [f for f in ucc_file if f'L-BFGS-B' in f]
ucc_data = pd.read_csv(os.path.join(UCC_folder,ucc_file[0]),sep='\t',header=None)
ucc_data[[1,2,3,4]].astype(float)
# ucc_data = ucc_data[ucc_data[0]!='random']
ucc_depth = list(ucc_data[3])
ucc_depth_std = list(ucc_data[4])

ax2.errorbar(x[:-1],ucc_depth[:-1],yerr=ucc_depth_std[:-1],marker='o',color='tab:blue',linestyle='none',label='UCC',zorder=5)
ax2.errorbar(x[-1],ucc_depth[-1:],yerr=ucc_depth_std[-1:],marker='o',color='tab:blue',linestyle='none',zorder=5)


adapt_folder = (f'./outputs/{nucleus.name}/v_performance/ADAPT')
adapt_file = os.listdir(adapt_folder)
adapt_file = [f for f in adapt_file if 'basis.csv' in f]
adapt_data = pd.read_csv(os.path.join(adapt_folder,adapt_file[0]),sep='\t')
if 'Success' in adapt_data.columns:
    adapt_data=adapt_data[adapt_data['Success']=='SUCCESS']
adapt_depth = adapt_data['Gates']
ax2.scatter(x[:-1],adapt_depth,marker='p',color=colors[1],edgecolors='black',zorder=5)

adapt_folder = (f'./outputs/{nucleus.name}/v_performance/ADAPT')
adapt_file = os.listdir(adapt_folder)
adapt_file = [f for f in adapt_file if 'random.csv' in f]
adapt_data = pd.read_csv(os.path.join(adapt_folder,adapt_file[0]),sep='\t')
if 'Success' in adapt_data.columns:
    adapt_data=adapt_data[adapt_data['Success']=='SUCCESS']
adapt_depth = adapt_data['Gates']
depth_mean = adapt_depth.mean()
std =adapt_depth.std()
ax2.errorbar(x[-1],depth_mean,std,marker='p',color=colors[1],linestyle='none',zorder=5, label='ADAPT')



ax1.set_xlabel(r'Basis Slater determinants ($v_{n}$)')
ax1.set_ylabel('Infidelity')
fig.legend(loc=(0.5,0.35),framealpha=1, frameon=True,edgecolor='black',fancybox=False, fontsize=12)
ax2.set_ylabel('Total operations')
ax2.set_ylim(0,600000)
ax1.set_xlim(-0.6,28.6)


fig.suptitle(r'Basis states as ref. states for UCC, $^{8}$B', fontsize=16)
try:
    os.makedirs('figures')
except OSError:
    pass

plt.show()

fig.savefig(f'./figures/{nucleus.name}/UCC_vs_ADAPT_{nucleus.name}.pdf',bbox_inches='tight')