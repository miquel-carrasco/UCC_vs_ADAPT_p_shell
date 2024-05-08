import numpy as np
from scipy import optimize
from tqdm import tqdm
from time import perf_counter
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import pandas as pd

from VQE.Nucleus import Nucleus
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz
from VQE.Utils import labels_all_combinations
from VQE.Methods import UCCVQE, OptimizationConvergedException, ADAPTVQE


def ADAPT_v_performance(tol_method = 'Manual') -> None:
    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)


    try:
        os.makedirs('outputs/v_performance/ADAPT')
    except OSError:
        pass

    
    output_folder = os.path.join('outputs/v_performance/ADAPT')
    methods = ['COBYLA', 'L-BFGS-B', 'BFGS', 'SLSQP']
    for method in methods:
        print(f'{method}')
        file = open(os.path.join(output_folder, f'{method}_performance_ADAPT.dat'), 'w')
        for n_v,v in tqdm(enumerate(vecs)):
            ref_state = v
            ADAPT_ansatz = ADAPTAnsatz(Li6, ref_state)
            vqe = ADAPTVQE(ADAPT_ansatz, method=method, tol_method=tol_method)
            vqe.run()
            if vqe.convergence:
                file.write(f'v{n_v}'+'\t'+f'{vqe.tot_operators}'+'\t'+'CONVERGED'+'\n')
            else:
                file.write(f'v{n_v}'+'\t'+f'{vqe.tot_operators}'+'\t'+'FAILED'+'\n')
        file.close()


def ADAPT_table(method: str = 'SLSQP', min_criterion:str = 'Repeated op', tol: float = 1e-10, tol_method: str = 'Manual') -> None:
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
                       min_criterion = min_criterion,
                       tol = tol,
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


def ADAPT_all_v_tracks(method: str = 'SLSQP', min_criterion: str = 'Repeated op', tol: float = 1e-10) -> None:
    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)

    for i,v in enumerate(vecs):
        ansatz = ADAPTAnsatz(Li6, v)
        vqe = ADAPTVQE(ansatz, method=method, min_criterion=min_criterion)
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

def ADAPT_plain_test(n_v: int = 0,
                     method: str = 'SLSQP',
                     min_criterion: str = 'Repeated op',
                     ftol: float = 1e-10,
                     gtol: float = 1e-10,
                     rhoend: float = 1e-10,
                     max_layers: int = 15) -> None:
    Li6 = Nucleus('Li6', 1)
    ref_state = np.eye(Li6.d_H)[n_v]
    ansatz = ADAPTAnsatz(Li6, ref_state)
    vqe = ADAPTVQE(ansatz,
                   method=method,
                   min_criterion=min_criterion,
                   ftol=ftol,
                   gtol=gtol,
                   rhoend=rhoend,
                   max_layers=max_layers)
    vqe.run()
    rel_error = vqe.rel_error
    energy = vqe.energy
    print((energy[-3]-energy[-2])/max([abs(energy[-2]),abs(energy[-3]),1]))
    print((energy[-2]-energy[-1])/max([abs(energy[-1]),abs(energy[-2]),1]))


def ADAPT_plot_evolution():

    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)

    try:
        os.makedirs('figures/ADAPT_evolution')
    except OSError:
        pass

    for j,v in enumerate(vecs):    
        ansatz = ADAPTAnsatz(Li6, v)
        vqe = ADAPTVQE(ansatz, method='SLSQP', min_criterion='None')
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


def ADAPT_tol_test(n_v: int):
    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)
    tols = np.logspace(-14, 0, 8)
    
    for tol in tols:
        ansatz = ADAPTAnsatz(Li6, vecs[n_v])
        vqe = ADAPTVQE(ansatz, method='SLSQP', min_criterion='None', tol=tol, tol_method= 'Manual')
        vqe.run()
        plt.plot(vqe.fcalls, vqe.rel_error, label=f'{tol}')
    
    ansatz = ADAPTAnsatz(Li6, vecs[n_v])
    vqe = ADAPTVQE(ansatz, method='SLSQP', min_criterion='None', tol_method= 'Automatic')
    vqe.run()
    plt.plot(vqe.fcalls, vqe.rel_error, label='Automatic')
    plt.yscale('log')
    plt.xlabel('Function calls')
    plt.xlim(0, 600)
    plt.ylabel('Relative error')
    plt.legend()
    plt.show()


def Gradient_evolution(method: str = 'SLSQP',
                       test_threshold: float = 1e-6,
                       stop_at_threshold: bool = True,
                       tol: float = 1e-10,
                       tol_method: str = 'Manual',
                       min_criterion: str = 'None',
                       max_layers: int = 15) -> None:
    
    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)

    fig, (ax0,ax1) = plt.subplots(2,1, figsize=(5,5), sharex=True)

    for i,v in enumerate(vecs):
        ADAPT_ansatz = ADAPTAnsatz(Li6, ref_state=v)

        vqe = ADAPTVQE(ADAPT_ansatz,
                       method=method,
                       return_data=True,
                       test_threshold=test_threshold,
                       stop_at_threshold=stop_at_threshold,
                       tol=tol,
                       tol_method=tol_method,
                       min_criterion=min_criterion,
                       max_layers=max_layers)
        
        result=vqe.run()

        gradients = result[0]
        energies = result[2]
        layers=np.linspace(1,len(gradients),len(gradients))
        print(([0.]+layers))
        ax0.plot(layers,(gradients))
        ax1.plot(([0]+layers),energies)
        for i,op in enumerate(ADAPT_ansatz.added_operators):
            if i>0:
                if ADAPT_ansatz.added_operators[i-1]==op:            
                    ax0.plot([i,(i+1)],[gradients[i-1],gradients[i]], color='red')
    ax0.set_yscale('log')
    ax0.set_xlim(0, max_layers)
    ax0.set_xlabel('Layer')
    ax0.set_ylabel('Gradient')

    plt.show()
