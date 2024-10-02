import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.linalg as la
from VQE.Nucleus import Nucleus, TwoBodyExcitationOperator
import pandas as pd
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz
from tqdm import tqdm

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

nuc_list=['Li6','Li8','B8','Li10','N10','B10','Be6','He6','Be8','Be10','C10']
x=[8.5,27.25,28.75,10,11.5,84,4.25,5.75,49.5,51,52.5]
colors=['tab:green','tab:red','tab:red','tab:green','tab:green','tab:brown','tab:blue','tab:blue','tab:orange','tab:orange','tab:orange']

fig,ax = plt.subplots(1,2,figsize=(13,6))

all_data = []
for i,nuc_name in tqdm(enumerate(nuc_list)):
    nuc = Nucleus(nuc_name,1)
    d_H = nuc.d_H
    ucc_ansatz=UCCAnsatz(nuc,ref_state=np.eye(d_H)[0],pool_format='ReducedII')

    data_nuc = {'Nucleus': nuc.name, 'Dimension': d_H, 'UCC_depth': 0, 'ADAPT_depth': 0, 
                'UCC_depth_std': 0, 'ADAPT_depth_std': 0,
                'UCC_layers': len(ucc_ansatz.operator_pool), 'ADAPT_layers': 0, 'ADAPT_layers_std': 0}
    UCC_folder = (f'./outputs/{nuc.name}/v_performance/UCC_Reduced')
    ucc_file = os.listdir(UCC_folder)
    ucc_file = [f for f in ucc_file if f'L-BFGS-B' in f]
    ucc_data = pd.read_csv(os.path.join(UCC_folder,ucc_file[0]),sep='\t',header=None)
    ucc_data[[1,2,3,4]].astype(float)
    ucc_data = ucc_data[ucc_data[0]=='v0']
    ucc_depth = ucc_data[3].iloc[0]
    ucc_depth_std = ucc_data[4].iloc[0]
    data_nuc['UCC_depth'] = ucc_depth
    data_nuc['UCC_depth_std'] = ucc_depth_std
    
    ax[0].errorbar(x[i],ucc_depth,yerr=ucc_depth_std,fmt='o',color=colors[i],markersize=10)

    adapt_folder = (f'./outputs/{nuc.name}/v_performance/ADAPT')
    adapt_file = os.listdir(adapt_folder)
    adapt_file = [f for f in adapt_file if 'basis.csv' in f]
    adapt_data = pd.read_csv(os.path.join(adapt_folder,adapt_file[0]),sep='\t')
    if 'Success' in adapt_data.columns:
        adapt_data=adapt_data[adapt_data['Success']=='SUCCESS']
    adapt_depth = adapt_data['Gates'].mean()
    adapt_depth_std = adapt_data['Gates'].std()
    adapt_layers = adapt_data['Layers'].mean()
    adapt_layers_std = adapt_data['Layers'].std()
    data_nuc['ADAPT_depth'] = adapt_depth
    data_nuc['ADAPT_depth_std'] = adapt_depth_std
    data_nuc['ADAPT_layers'] = adapt_layers
    data_nuc['ADAPT_layers_std'] = adapt_layers_std
    ax[0].errorbar(x[i],adapt_depth,yerr=adapt_depth_std,marker='p',color=colors[i],markersize=10)
    all_data.append(data_nuc)


ax[0].vlines([5,10,28,51,84],0,1e7,linestyles='dashed',color='black',alpha=0.5)

ax[0].errorbar([],[],[],fmt='o',label='UCC',color='grey')
ax[0].errorbar([],[],[],fmt='p',label='ADAPT',color='grey')

ax[0].text(5,10,r'$d_{\mathcal{H}}=5$',rotation=90,va='top',ha='right',fontsize=11)
ax[0].text(10,10,r'$d_{\mathcal{H}}=10$',rotation=90,va='top',ha='right',fontsize=11)
ax[0].text(28,10,r'$d_{\mathcal{H}}=28$',rotation=90,va='top',ha='right',fontsize=11)
ax[0].text(51,10,r'$d_{\mathcal{H}}=51$',rotation=90,va='top',ha='right',fontsize=11)
ax[0].text(84,10,r'$d_{\mathcal{H}}=84$',rotation=90,va='top',ha='right',fontsize=11)


ax[0].set_xlabel(r'$dim(\mathcal{H})$')
ax[0].set_ylabel('Total operations')
ax[0].set_yscale('log')
ax[0].set_ylim(1,1e7)

ax[0].legend(loc=(0.7,0.4),framealpha=1, frameon=True,edgecolor='black',fancybox=False)

df=pd.DataFrame(all_data)

df.to_csv('./outputs/all_nuclei_ADAPT_UCC.csv',sep='\t',index=False)
for n,nuc in enumerate(nuc_list):
    row = df[df['Nucleus']==nuc]
    ax[1].errorbar(x[n], row['ADAPT_layers'], row['ADAPT_layers_std'],marker='p', color=colors[n],markersize=10)
    ax[1].errorbar(x[n], row['UCC_layers'], marker='o', color=colors[n],markersize=10)


ax[1].vlines([5,10,28,51,84],0,1000,linestyles='dashed',color='black',alpha=0.5)

ax[1].set_xlabel(r'$dim(\mathcal{H})$')
ax[1].set_ylabel(r'\emph{Ansatz} layers')
ax[1].set_ylim(0,180)

ax[1].text(5,100,r'$d_{\mathcal{H}}=5$',rotation=90,va='top',ha='right',fontsize=11)
ax[1].text(10,100,r'$d_{\mathcal{H}}=10$',rotation=90,va='top',ha='right',fontsize=11)
ax[1].text(28,100,r'$d_{\mathcal{H}}=28$',rotation=90,va='top',ha='right',fontsize=11)
ax[1].text(51,100,r'$d_{\mathcal{H}}=51$',rotation=90,va='top',ha='right',fontsize=11)
ax[1].text(84,100,r'$d_{\mathcal{H}}=84$',rotation=90,va='top',ha='right',fontsize=11)


fig.savefig(f'./figures/all_nuclei_ADAPT_UCC.pdf', bbox_inches='tight')