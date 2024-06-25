import numpy as np
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
         'xtick.labelsize': 11,
         'ytick.labelsize': 14,
         "text.usetex": True,
         "font.family": "serif",
         "font.serif": ["Palatino"]
         }
plt.rcParams.update(params)

Li6 = Nucleus('Li6',1)
d_H = Li6.d_H

runs=100


outputs_folder=(f'./outputs/{Li6.name}/v_performance/UCC_Reduced')
files=os.listdir(outputs_folder)
files=[f for f in files if f'ntimes={runs}' in f]

optimizers=[]
dodge=np.linspace(-0.3,0.3,len(files))
markers=['o','v','p','d','s','X']

fig, ax1 = plt.subplots()
ax2=ax1.twinx()

infidelity=[]
for i in range(d_H):
    vi=np.eye(d_H)[:,i]
    overlapp=np.abs(Li6.eig_vec[:,0].conj().T@vi)**2
    infidelity.append(1-overlapp)
ax1.bar(np.arange(d_H),infidelity,alpha=0.5,color='grey',label='Infidelity',zorder=1,edgecolor='black')


v=[r'$|\frac{1}{2}, -\frac{1}{2},\frac{1}{2}, \frac{1}{2}\rangle$',
    r'$|\frac{1}{2}, -\frac{1}{2},\frac{3}{2}, \frac{1}{2}\rangle$',
    r'$|\frac{1}{2}, \frac{1}{2},\frac{1}{2}, -\frac{1}{2}\rangle$',
    r'$|\frac{1}{2}, \frac{1}{2},\frac{3}{2}, -\frac{1}{2}\rangle$',
    r'$|\frac{3}{2}, -\frac{3}{2},\frac{3}{2}, \frac{3}{2}\rangle$',
    r'$|\frac{3}{2}, -\frac{1}{2},\frac{1}{2}, \frac{1}{2}\rangle$',
    r'$|\frac{3}{2}, -\frac{1}{2},\frac{3}{2}, \frac{1}{2}\rangle$',
    r'$|\frac{3}{2}, \frac{1}{2},\frac{1}{2}, -\frac{1}{2}\rangle$',
    r'$|\frac{3}{2}, \frac{1}{2},\frac{3}{2}, -\frac{1}{2}\rangle$',
    r'$|\frac{3}{2}, \frac{3}{2},\frac{3}{2}, -\frac{3}{2}\rangle$',
    'Randomized']

for i,f in enumerate(files):
    file_path=os.path.join(outputs_folder,f)
    data=open(file_path,'r').readlines()
    data=[d.strip('\n').split('\t') for d in data]
    optimizer=f.split('_')[0]
    optimizers.append(optimizer)
    mean=[]
    std=[]
    for j,d in enumerate(data):          
        mean.append(float(d[1]))
        std.append(float(d[2]))
    x=np.arange(len(v))+dodge[i]
    ax2.errorbar(x,mean,yerr=std,marker=markers[i],linestyle='none',label=f.split('_')[0],zorder=5)






ax1.set_xticks(np.arange(len(v)),v,rotation=45)
ax1.set_title(f'Optimizers and vectors performance (randomized $t_0$ {runs} runs)')
ax1.set_xlabel(r'State ($|j_p, m_p,j_n, m_n\rangle$)')
ax1.set_ylabel('Infidelity')
fig.legend(loc=(0.2,0.67),framealpha=1, frameon=True,edgecolor='black',fancybox=False)
ax2.set_ylabel('Function calls')
ax2.set_ylim(0,755)

try:
    os.makedirs('figures')
except OSError:
    pass


fig.savefig(f'figures/opt_vect_performance_random_{runs}runs.pdf',bbox_inches='tight')