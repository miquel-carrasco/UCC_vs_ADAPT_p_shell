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

nucleus = Nucleus('B8',1)
d_H = nucleus.d_H

runs=50


UCC_folder=(f'./outputs/{nucleus.name}/v_performance/UCC_Reduced')
UCC_files=os.listdir(UCC_folder)
UCC_files=[f for f in UCC_files if f'L-BFGS-B_ntimes={runs}' in f]

ADAPT_folder=(f'./outputs/{nucleus.name}/v_performance/ADAPT')
ADAPT_files=os.listdir(ADAPT_folder)
ADAPT_files=[f for f in ADAPT_files if f'.dat' in f]
# optimizers=[]
# dodge=np.linspace(-0.3,0.3,len(UCC_files))
# markers=['o','v','p','d','s','X']
colors=['tab:blue','tab:orange','tab:green','tab:red']

fig, ax1 = plt.subplots()
ax2=ax1.twinx()

infidelity=[]
vec_list = [0,1,2,3,4,5,6,7,8,18,19,21,22,23,27]
for i in vec_list:
    vi = np.eye(d_H)[:,i]
    overlapp=np.abs(nucleus.eig_vec[:,0].conj().T@vi)**2
    infidelity.append(1-overlapp)
ax1.bar(np.arange(len(vec_list)),infidelity,alpha=0.5,color='grey',label='Infidelity',zorder=1,edgecolor='black')

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
    data = data[:-1]
    optimizer=f.split('_')[0]
    # optimizers.append(optimizer)
    mean=[]
    std=[]
    for j,d in enumerate(data):          
        mean.append(float(d[1]))
        std.append(float(d[2]))
    x=np.arange(15)
    ax2.errorbar(x,mean,yerr=std,marker='o',color=colors[i],linestyle='none',label='UCC',zorder=5)

    ADAPT_f = ADAPT_files[i]
    data=open(os.path.join(ADAPT_folder,ADAPT_f),'r').readlines()
    data=[d.strip('\n').split('\t') for d in data]
    data = [d for d in data if int(d[0][1:]) in vec_list]
    converged = []
    x_c = []
    failed = []
    x_f = []
    for j,d in enumerate(data):
        if d[3] == 'SUCCESS':
            converged.append(float(d[1]))
            x_c.append(x[j])
        else:
            failed.append(float(d[1]))
            x_f.append(x[j])
    ax2.scatter(x_c,converged,marker='^',color=colors[1],edgecolors='black',zorder=5, label='ADAPT')
    ax2.scatter(x_f,failed,marker='X',color=colors[1],edgecolors='black',zorder=5)



# ax1.set_xticks(np.arange(len(v)),v,rotation=45)
ax1.set_xticks(np.arange(len(vec_list)),[f'v{i}' for i in vec_list])
ax1.set_title(f'UCC vs ADAPT')
ax1.set_xlabel(r'State ($|j_p, m_p,j_n, m_n\rangle$)')
ax1.set_ylabel('Infidelity')
fig.legend(loc=(0.1,0.75),framealpha=1, frameon=True,edgecolor='black',fancybox=False)
ax2.set_ylabel('Gates')


try:
    os.makedirs('figures')
except OSError:
    pass


fig.savefig(f'./figures/{nucleus.name}/UCC_vs_ADAPT.pdf',bbox_inches='tight')