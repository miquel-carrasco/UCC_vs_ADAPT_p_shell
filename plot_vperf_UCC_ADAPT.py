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


UCC_folder=('./outputs/v_performance/UCC')
UCC_files=os.listdir(UCC_folder)
UCC_files=[f for f in UCC_files if f'ntimes={runs}' in f]

ADAPT_folder=('./outputs/v_performance/ADAPT')
ADAPT_files=os.listdir(ADAPT_folder)
optimizers=[]
dodge=np.linspace(-0.3,0.3,len(UCC_files))
markers=['o','v','p','d','s','X']
colors=['tab:blue','tab:orange','tab:green','tab:red']

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
    r'$|\frac{3}{2}, \frac{3}{2},\frac{3}{2}, -\frac{3}{2}\rangle$']


for i,f in enumerate(UCC_files):
    file_path=os.path.join(UCC_folder,f)
    data=open(file_path,'r').readlines()
    data=[d.strip('\n').split('\t') for d in data]
    optimizer=f.split('_')[0]
    optimizers.append(optimizer)
    mean=[]
    std=[]
    for j,d in enumerate(data):          
        mean.append(float(d[1])*9)
        std.append(float(d[2])*9)
    x=np.arange(len(v))+dodge[i]
    ax2.errorbar(x,mean,yerr=std,marker=markers[i],color=colors[i],linestyle='none',label=f.split('_')[0],zorder=5)

    ADAPT_f = ADAPT_files[i]
    data=open(os.path.join(ADAPT_folder,ADAPT_f),'r').readlines()
    data=[d.strip('\n').split('\t') for d in data]
    converged = []
    x_c = []
    failed = []
    x_f = []
    for j,d in enumerate(data):
        if d[2] == 'SUCCESSED':
            converged.append(float(d[1]))
            x_c.append(x[j])
        else:
            failed.append(float(d[1]))
            x_f.append(x[j])
    ax2.scatter(x_c,converged,marker='^',color=colors[i],edgecolors='black',zorder=5)
    ax2.scatter(x_f,failed,marker='X',color=colors[i],edgecolors='black',zorder=5)



ax1.set_xticks(np.arange(len(v)),v,rotation=45)
ax1.set_title(f'Overall UCC vs ADAPT performance')
ax1.set_xlabel(r'State ($|j_p, m_p,j_n, m_n\rangle$)')
ax1.set_ylabel('Infidelity')
fig.legend(loc=(0.2,0.67),framealpha=1, frameon=True,edgecolor='black',fancybox=False)
ax2.set_ylabel('Gate count')
ax2.set_ylim(0,8010)

try:
    os.makedirs('figures')
except OSError:
    pass


fig.savefig(f'./figures/opt_vect_performance_random_{runs}runs_vs_ADAPT.pdf',bbox_inches='tight')