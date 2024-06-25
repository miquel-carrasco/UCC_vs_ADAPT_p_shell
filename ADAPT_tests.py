import numpy as np
from scipy import optimize
from tqdm import tqdm
from time import perf_counter
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import pandas as pd
from time import time
import concurrent.futures

from VQE.Nucleus import Nucleus
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz
from VQE.Utils import labels_all_combinations
from VQE.Methods import UCCVQE, OptimizationConvergedException, ADAPTVQE


def ADAPT_v_performance(nuc: str,
                        max_layers: int = 60,
                        conv_criterion: str = 'Repeated op',
                        test_threshold: float = 1e-6,
                        stop_at_threshold: bool = True) -> None:
    
    nucleus = Nucleus(nuc, 1)
    vecs = np.eye(nucleus.d_H)

    try:
        os.makedirs(f'outputs/{nuc}/v_performance/ADAPT')
    except OSError:
        pass

    output_folder = os.path.join(f'outputs/{nuc}/v_performance/ADAPT')
    methods = ['L-BFGS-B', 'BFGS', 'SLSQP']
    for method in methods:
        print(f'{method}')
        file = open(os.path.join(output_folder, f'{method}_performance_ADAPT.dat'), 'w')
        for n_v in tqdm(range(5)):
            ref_state = vecs[n_v]
            ADAPT_ansatz = ADAPTAnsatz(nucleus, ref_state, pool_format='Reduced')

            vqe = ADAPTVQE(ADAPT_ansatz,
                           method = method,
                           max_layers = max_layers,
                           conv_criterion = conv_criterion,
                           test_threshold = test_threshold,
                           stop_at_threshold = stop_at_threshold)
            vqe.run()
            if vqe.success:
                file.write(f'v{n_v}'+'\t'+f'{vqe.tot_operators}'+'\t'+'SUCCESSED'+'\n')
            else:
                file.write(f'v{n_v}'+'\t'+f'{vqe.tot_operators}'+'\t'+'FAILED'+'\n')
        file.close()


def ADAPT_table(method: str = 'SLSQP',
                conv_criterion:str = 'Repeated op',
                ftol: float = 1e-10,
                gtol: float = 1e-10,
                rhoend: float = 1e-10,
                max_layers: int = 15,
                tol_method: str = 'Manual') -> None:

    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)
    try:
        os.makedirs('outputs/ADAPT_table')
    except OSError:
        pass

    for i,v in enumerate(vecs):
        ansatz = ADAPTAnsatz(Li6, v)
        vqe = ADAPTVQE(ansatz,
                       method = method,
                       return_data = True,
                       conv_criterion = conv_criterion,
                       ftol = ftol,
                       tol_method = tol_method)
        result = vqe.run()
        data = {'Operator label': ['None'] + [str(op.label) for op in ansatz.added_operators],
                'Operator excitation': ['None'] + [str(op.ijkl) for op in ansatz.added_operators],
                'Gradient': ['None'] + [np.linalg.norm(grad) for grad in result[0]],
                'Final parameters gradient': ['None'] + result[1],
                'Energy': result[2],
                'Relative error': result[3],
                'Function calls': result[4]}
        df = pd.DataFrame(data)
        df.to_csv(f'outputs/ADAPT_table/v{i}_ADAPT_table.csv', sep='\t', index=False)


def ADAPT_all_v_tracks(method: str = 'SLSQP',
                       conv_criterion: str = 'Repeated op',
                       ftol: float = 1e-10,
                       gtol: float = 1e-10,
                       rhoend: float = 1e-10,
                       tol_method: str = 'Manual',
                       max_layers: int = 15) -> None:
    
    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)

    for i,v in enumerate(vecs):
        ansatz = ADAPTAnsatz(Li6, v)
        vqe = ADAPTVQE(ansatz,
                       method = method,
                       conv_criterion = conv_criterion,
                       ftol = ftol,
                       gtol = gtol,
                       rhoend = rhoend,
                       tol_method = tol_method,
                       max_layers = max_layers)
        
        result = vqe.run()
        rel_error = vqe.rel_error
        fcalls = vqe.fcalls
        plt.plot(fcalls, rel_error, label=f'$v_{i}$')
    plt.yscale('log')
    plt.xlabel('Function calls')
    plt.ylabel('Relative error')
    plt.xlim(0, 610)
    plt.legend()

    plt.show()


def ADAPT_plain_test(nuc: str,
                     n_v: int = 1,
                     method: str = 'SLSQP',
                     test_threshold: float = 1e-6,
                     stop_at_threshold: bool = True,
                     conv_criterion: str = 'Repeated op',
                     max_layers: int = 10,
                     pool_format: str = 'Reduced') -> None:
    
    nucleus = Nucleus(nuc, 1)
    ref_state = np.eye(nucleus.d_H)[n_v]
    ansatz = ADAPTAnsatz(nucleus, ref_state, pool_format= pool_format)

    vqe = ADAPTVQE(ansatz,
                   method=method,
                   test_threshold=test_threshold,
                   stop_at_threshold=stop_at_threshold,
                   conv_criterion=conv_criterion,
                   max_layers=max_layers)
    time1 = time()
    vqe.run()
    time2 = time()
    print(time2-time1)
    rel_error = vqe.rel_error
    energy = vqe.energy

    plt.plot(vqe.tot_operators_layers, rel_error)
    plt.yscale('log')
    plt.show()



def parameters_evolution(nuc: str,
                         method: str = 'SLSQP',
                         conv_criterion: str = 'Repeated op',
                         max_layers: int = 100) -> None:
    
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

    try:
        os.makedirs(f'figures/{nuc}/ADAPT_parameters_evolution')
    except OSError:
        pass

    nucleus = Nucleus(nuc, 1)

    for n_v in tqdm(range(4)):
        ref_state = np.eye(nucleus.d_H)[n_v]
        ansatz = ADAPTAnsatz(nucleus, ref_state)

        vqe = ADAPTVQE(ansatz,
                    method=method,
                    conv_criterion=conv_criterion,
                    max_layers=max_layers,
                    return_data=True)
                            
        result = vqe.run()

        num_layers = len(vqe.parameters)
        x = np.linspace(1, num_layers, len(vqe.parameters))
        #plt.vlines(x, ymin=-np.pi, ymax=np.pi, color='grey', linestyle='--',alpha=0.5)
        for i,parameter in enumerate(vqe.parameter_layers):
            plt.plot(x[-len(parameter):],parameter,'-o')
            if num_layers < 20:
                ijkl = ansatz.added_operators[i].ijkl
                plt.text(x[-len(parameter)]+0.3,1.6, r'$T^{%d,%d} _{%d,%d}$'%(ijkl[0],ijkl[1],ijkl[2],ijkl[3]), fontsize=9)
        plt.xlim(1, len(vqe.parameters)+1)
        #plt.ylim(-2, 2)
        plt.xlabel('Layer')
        plt.ylabel('Parameter value')
        plt.title(f'{nuc} parameters evolution for $v_{n_v}$ ({num_layers} layers)')
        plt.savefig(f'figures/{nuc}/ADAPT_parameters_evolution/v{n_v}_ADAPT_parameters_evolution.pdf', bbox_inches='tight')
        plt.close()


def ADAPT_plot_evolution(nuc: str,
                         method: str = 'SLSQP',
                         conv_criterion: str = 'Repeated op',
                         tol_method: str = 'Manual',
                         max_layers: int = 100) -> None:

    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)

    try:
        os.makedirs('figures/ADAPT_evolution')
    except OSError:
        pass

    for j,v in enumerate(vecs):    
        ansatz = ADAPTAnsatz(Li6, v)

        vqe = ADAPTVQE(ansatz, method='SLSQP',
                       conv_criterion='None',
                       tol_method='Manual',
                       max_layers=15)

        vqe.run()
        state_layers = vqe.state_layers

        nrows = int(np.ceil(np.sqrt(len(state_layers))))
        ncols = int(np.ceil(len(state_layers) / nrows))

        fig, axs = plt.subplots(nrows, ncols, figsize=(8,8))
        plt.subplots_adjust(hspace=0.5)

        for i, state in enumerate(state_layers):
            row = i // ncols
            col = i % ncols
            axs[row, col].bar(range(len(state)), np.abs(state))
            if i == 0:
                axs[row, col].set_title('Reference state')
            else:
                ijkl = ansatz.added_operators[i-1].ijkl
                axs[row, col].set_title(r'$T^{%d \; %d}_{%d \; %d}$'%(ijkl[0], ijkl[1], ijkl[2], ijkl[3]))
        
        fig.savefig(f'figures/ADAPT_evolution/v{j}_ADAPT_evolution.pdf', bbox_inches='tight')
        plt.close()


def tol_test(n_v: int,
                   conv_criterion: str = 'None',
                   max_layers: int = 15,
                   test_threshold: float = 1e-6,
                   stop_at_threshold: bool = True) -> None:
    
    params = {'axes.linewidth': 1.4,
            'axes.labelsize': 14,
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
    
    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)
    ftols = np.logspace(-8, -2, 7)
    
    fig, ((ax1, ax2),( ax3, ax4)) = plt.subplots(2,2, figsize=(8,6), sharey=True)

    for ftol in ftols:
        ansatz = ADAPTAnsatz(Li6, vecs[n_v], pool_format='Reduced')
        vqe = ADAPTVQE(ansatz,
                       method = 'SLSQP',
                       conv_criterion = conv_criterion,
                       ftol = ftol,
                       gtol = 0.0, 
                       rhoend = 0.0,
                       tol_method = 'Manual',
                       max_layers = max_layers,
                       test_threshold=test_threshold,
                       stop_at_threshold = stop_at_threshold)
        vqe.run()
        if n_v in [0,2,4,6,8,9]:
            rel_error = abs((vqe.energy-Li6.eig_val[1])/Li6.eig_val[1])
        else:
            rel_error = vqe.rel_error
        ax1.plot(vqe.tot_operators_layers, rel_error, label=f'{ftol}')
        ax1.set_yscale('log')
        ax1.set_title('ftol (SLSQP)')
        ax1.legend(fontsize='small',title='ftol')
        ax1.set_ylabel('Relative error')


    rftols = np.logspace(-8, -2, 7)
    for rftol in rftols:
        ansatz = ADAPTAnsatz(Li6, vecs[n_v], pool_format='Reduced')
        vqe = ADAPTVQE(ansatz,
                       method = 'L-BFGS-B',
                       conv_criterion = conv_criterion,
                       ftol = rftol,
                       gtol = 0.0,
                       rhoend = 0.0, 
                       tol_method = 'Manual',
                       max_layers = max_layers,
                       test_threshold=test_threshold,
                       stop_at_threshold = stop_at_threshold)
        vqe.run()
        if n_v in [0,2,4,6,8,9]:
            rel_error = abs((vqe.energy-Li6.eig_val[1])/Li6.eig_val[1])
        else:
            rel_error = vqe.rel_error
        ax2.plot(vqe.tot_operators_layers, rel_error, label=f'{rftol}')
        ax2.set_yscale('log')   
        ax2.set_title('rftol (L-BFGS-B)') 
        ax2.legend(fontsize='small',title='rftol')

    gtols = np.logspace(-7, -1, 7)
    for gtol in gtols:
        ansatz = ADAPTAnsatz(Li6, vecs[n_v], pool_format='Reduced')
        vqe = ADAPTVQE(ansatz,
                       method = 'BFGS',
                       conv_criterion = conv_criterion,
                       ftol = 0.0,
                       gtol = gtol,
                       rhoend = 0.0, 
                       tol_method = 'Manual',
                       max_layers = max_layers,
                       test_threshold=test_threshold,
                       stop_at_threshold = stop_at_threshold)
        vqe.run()
        if n_v in [0,2,4,6,8,9]:
            rel_error = abs((vqe.energy-Li6.eig_val[1])/Li6.eig_val[1])
        else:
            rel_error = vqe.rel_error
        ax3.plot(vqe.tot_operators_layers, rel_error, label=f'{gtol}')
        ax3.set_yscale('log')
        ax3.set_title('gtol (BFGS)')
        ax3.legend(fontsize='small',title='gtol')
        ax3.set_xlabel('Operators used')
        ax3.set_ylabel('Relative error')

    rhos = np.logspace(-8, -2, 7)
    for rho in rhos:
        ansatz = ADAPTAnsatz(Li6, vecs[n_v], pool_format='Reduced')
        vqe = ADAPTVQE(ansatz,
                       method = 'COBYLA',
                       conv_criterion = conv_criterion,
                       ftol = 0.0,
                       gtol = 0.0,
                       rhoend = rho, 
                       tol_method = 'Manual',
                       max_layers = max_layers,
                       test_threshold=test_threshold,
                       stop_at_threshold = stop_at_threshold)
        vqe.run()
        if n_v in [0,2,4,6,8,9]:
            rel_error = abs((vqe.energy-Li6.eig_val[1])/Li6.eig_val[1])
        else:
            rel_error = vqe.rel_error
        ax4.plot(vqe.tot_operators_layers, rel_error, label=f'{rho}')
        ax4.set_yscale('log')
        ax4.set_title('rhoend (COBYLA)')
        ax4.legend(fontsize='small',title='rhoend')
        ax4.set_xlabel('Operators used')

    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(f'ADAPT tolerance test for $v_{n_v}$', fontsize=16)

    fig.savefig(f'figures/ADAPT_tol_test_{n_v}.pdf', bbox_inches='tight')     


def Gradient_evolution(method: str = 'SLSQP',
                       test_threshold: float = 1e-6,
                       stop_at_threshold: bool = True,
                       ftol: float = 1e-10,
                       gtol: float = 1e-10,
                       tol_method: str = 'Manual',
                       conv_criterion: str = 'None',
                       max_layers: int = 15) -> None:
    
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

    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)

    fig, ax = plt.subplots(2,1, figsize=(4,6), sharex=True)
    a=0
    for i,v in enumerate(vecs):
        print(f'v_{i}')
        ADAPT_ansatz = ADAPTAnsatz(Li6, ref_state=v)

        vqe = ADAPTVQE(ADAPT_ansatz,
                       method=method,
                       return_data=True,
                       test_threshold=test_threshold,
                       stop_at_threshold=stop_at_threshold,
                       ftol=ftol,
                       gtol=gtol,
                       tol_method=tol_method,
                       conv_criterion=conv_criterion,
                       max_layers=max_layers)
        
        result=vqe.run()

        gradients = result[0]
        energies = result[2]
        if i in [1,3,5,7]:
            rel_errors = result[3]
        else:
            rel_errors = abs((energies-Li6.eig_val[1])/Li6.eig_val[1])
        layers=np.linspace(1,len(gradients)+1,(len(gradients)+1))
        ax[0].plot(layers[:-1],gradients)
        ax[1].plot(layers,rel_errors,label=f'$v_{i}$')
        for j in range(1,(len(layers)-1)):
            if ADAPT_ansatz.added_operators[j-1]==ADAPT_ansatz.added_operators[j]:
                a+=1
                label = 'Repeated op' if a==1 else None
                ax[0].plot([j,(j+1)],[gradients[j-1],gradients[j]], color='red',label=label)


    for i in range(1,len(layers)+1,2):
        ax[0].axvline(x=i, color='grey', linestyle='--', alpha=0.5)
        ax[1].axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_xlim(1, max_layers)
    ax[0].set_ylabel('Gradient')
    ax[1].set_xlabel('Layer')
    ax[1].set_ylabel('Relative error')
    ax[0].legend(fontsize='small')
    ax[1].legend(fontsize='small')
    
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0.3)

    try:
        os.makedirs('figures/ADAPT_Gradient_evolution')
    except OSError:
        pass

    fig.savefig(f'figures/ADAPT_Gradient_evolution/Gradient_evolution_{method}_tols={ftol}_{gtol}.pdf', bbox_inches='tight')


def pool_format_test(nuc: str,
                     n_v: int = 1,
                     method: str = 'SLSQP',
                     test_threshold: float = 1e-6,
                     stop_at_threshold: bool = True,
                     conv_criterion: str = 'None',
                     max_layers: int = 100) -> None:
    
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

    nucleus = Nucleus(nuc, 1)
    ref_state = np.eye(nucleus.d_H)[n_v]
    pool_formats = ['All', 'Only acting', 'Reduced']
    for pool_format in pool_formats:
        time1 = time()
        ansatz = ADAPTAnsatz(nucleus, ref_state, pool_format=pool_format)
        print(f'{pool_format} --> {len(ansatz.operator_pool)} operators')
        vqe = ADAPTVQE(ansatz,
                       method=method,
                       test_threshold=test_threshold,
                       stop_at_threshold=stop_at_threshold,
                       conv_criterion=conv_criterion,
                       max_layers=max_layers)
        vqe.run()
        rel_error = vqe.rel_error
        energy = vqe.energy
        time2 = time()
        print(time2-time1)

        plt.plot(vqe.tot_operators_layers, rel_error, label=f'{pool_format}')
    plt.yscale('log')
    plt.xlabel('Operators used')
    plt.ylabel('Relative error')
    plt.legend()
    plt.show()


def one_step_test(nuc: str,
                  method: str = 'SLSQP',
                  test_threshold: float = 1e-6,
                  stop_at_threshold: bool = True,
                  tol_method: str = 'Manual',
                  conv_criterion: str = 'Repeated',
                  pool_format: str = 'Reduced',
                  max_layers: int = 100,
                  one_step_layers: list = [100]) -> None:
    
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
    # plt.rcParams.update(params)

    nucleus = Nucleus(nuc, 1)

    for n in range(15):
        ref_state = np.random.uniform(-1,1,nucleus.d_H)
        ref_state = ref_state/np.linalg.norm(ref_state)
        # ref_state = np.eye(nucleus.d_H)[1]

        ansatz = ADAPTAnsatz(nucleus, ref_state, pool_format=pool_format)
        vqe = ADAPTVQE(ansatz,
                    method=method,
                    test_threshold=test_threshold,
                    stop_at_threshold=stop_at_threshold,
                    tol_method=tol_method,
                    conv_criterion=conv_criterion)
        vqe.run()
        label = f'ADAPT ({len(vqe.parameters)} layers)' if n==0 else None
        plt.plot(vqe.tot_operators_layers, vqe.rel_error, label=label, color='blue', alpha = 0.5)


        ansatz = ADAPTAnsatz(nucleus, ref_state, pool_format=pool_format)
        vqe = ADAPTVQE(ansatz,
                    method=method,
                    test_threshold=test_threshold,
                    stop_at_threshold=stop_at_threshold,
                    tol_method=tol_method,
                    conv_criterion=conv_criterion,
                    max_layers=max_layers)
        vqe.run_one_step(final_run=False)
        label = f'Seq-ADAPT ({len(vqe.parameters)} layers)' if n==0 else None
        plt.plot(vqe.tot_operators_layers, vqe.rel_error, color = 'red', alpha = 0.5, label=label)


        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        for i,max_layers in enumerate(one_step_layers):
            ansatz = ADAPTAnsatz(nucleus, ref_state, pool_format=pool_format)
            vqe = ADAPTVQE(ansatz,
                        method=method,
                        test_threshold=test_threshold,
                        stop_at_threshold=stop_at_threshold,
                        tol_method=tol_method,
                        conv_criterion=conv_criterion,
                        max_layers=max_layers)
            vqe.run_one_step(final_run=True)
            label = f'Seq-ADAPT + final run ({max_layers} layers)' if n==0 else None
            plt.plot(vqe.tot_operators_layers, vqe.rel_error, label=label, color=colors[i], alpha = 0.5) 

    plt.yscale('log')
    plt.xlabel('Circuit depth')
    plt.ylabel('Relative error')
    plt.legend(fontsize='small')
    plt.title(f'ADAPT vs Seq-ADAPT ({method})')

    plt.savefig(f'figures/{nuc}/Seq_ADAPT_{method}_vrandom.pdf')
    print(f'{nuc} done!')

if __name__ == '__main__':
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

    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)

    ansatz_0 = ADAPTAnsatz(Li6, vecs[0], pool_format='Reduced')
    ansatz_1 = ADAPTAnsatz(Li6, vecs[1], pool_format='Reduced')

    VQE_0 = ADAPTVQE(ansatz_0, method='L-BFGS-B', conv_criterion='Repeated', max_layers=15)
    VQE_1 = ADAPTVQE(ansatz_1, method='L-BFGS-B', conv_criterion='Repeated', max_layers=15)

    VQE_0.run()
    VQE_1.run()

    plt.plot(VQE_0.fcalls, VQE_0.rel_error, label=r'$|\frac{1}{2}-\frac{1}{2}\frac{1}{2}\frac{1}{2}\rangle$')
    plt.plot(VQE_1.fcalls, VQE_1.rel_error, label=r'$|\frac{1}{2}-\frac{1}{2}\frac{3}{2}\frac{1}{2}\rangle$')

    plt.hlines(abs((Li6.eig_val[0]-Li6.eig_val[1])/Li6.eig_val[0]),0,1000, color='grey', linestyle='--', alpha=0.5)
    plt.text(450, abs((Li6.eig_val[0]-Li6.eig_val[1])/Li6.eig_val[0]), '1st excited state energy', verticalalignment='bottom', horizontalalignment='right')

    plt.yscale('log')
    plt.xlim(0, 500)
    plt.xlabel('Function calls')
    plt.ylabel('Relative error')
    plt.legend(loc=(0.25,0.25), fontsize='large')

    plt.savefig('figures/Li6/ADAPT_test_2basis.pdf', bbox_inches='tight')
