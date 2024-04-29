import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from time import perf_counter
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import pandas as pd

from VQE.Nucleus import Nucleus
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz, ADAPTAnsatzNoTrotter
from VQE.Utils import labels_all_combinations
from VQE.Methods import UCCVQE, OptimizationConvergedException, ADAPTVQE


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


def UCC_vs_ADAPT(n_v: int) -> None:
    Li6 = Nucleus('Li6', 1)
    ref_state = np.eye(Li6.d_H)[n_v]
    
    ADAPT_ansatz = ADAPTAnsatz(Li6, ref_state)
    ADAPT_vqe = ADAPTVQE(ADAPT_ansatz, method='SLSQP',min_criterion='None')
    ADAPT_vqe.run()
    adapt_rel_error = ADAPT_vqe.rel_error
    adapt_fcalls = ADAPT_vqe.fcalls

    UCC_ansatz = UCCAnsatz(Li6, ref_state)
    init_param = np.zeros(len(UCC_ansatz.operators))
    UCC_vqe = UCCVQE(UCC_ansatz,init_param=init_param, method='SLSQP')
    UCC_vqe.run()
    ucc_rel_error = UCC_vqe.rel_error
    ucc_fcalls = UCC_vqe.fcalls

    print(ADAPT_vqe.tot_operators_layers)
    print([i*9 for i in UCC_vqe.fcalls])

    plt.plot(adapt_fcalls, adapt_rel_error, label='ADAPT-VQE')
    plt.plot(ucc_fcalls, ucc_rel_error, label='UCC-VQE')
    plt.vlines(ADAPT_vqe.layer_fcalls, colors='lightgrey',ymin=1e-8,ymax=10, linestyles='dashed')
    plt.ylim(1e-7, 10)
    plt.xlim(0, 500)
    plt.yscale('log')
    plt.xlabel('Function calls')
    plt.ylabel('Relative error')
    plt.legend()
    plt.title(f'UCC vs ADAPT-VQE (v{n_v})')

    plt.show()


def UCC_v_performance(method: str, n_times: int) -> None:
    Li6 = Nucleus('Li6', 1)
    vecs = np.eye(Li6.d_H)

    try:
        os.makedirs('outputs/v_performance/UCC')
    except OSError:
        pass
    output_folder = os.path.join('outputs/v_performance/UCC')


    file = open(os.path.join(output_folder, f'{method}_performance_randomt0_ntimes={n_times}.dat'), 'w')
    for n_v,v in enumerate(vecs):
        print(f'{n_v}')
        ref_state = v
        UCC_ansatz = UCCAnsatz(Li6, ref_state)

        calls = 0
        calls2 = 0
        n_fails = 0
        for n in tqdm(range(n_times)):
            init_param = np.random.uniform(low=-np.pi, high=np.pi,size=len(UCC_ansatz.operators))
            random.shuffle(UCC_ansatz.operators)
            vqe = UCCVQE(UCC_ansatz, init_param=init_param, method=method)
            vqe.run()
            if vqe.convergence:
                calls += vqe.fcalls[-1]
                calls2 += vqe.fcalls[-1]**2
            else:
                n_fails += 1
        mean_calls = calls/(n_times-n_fails)
        std_calls = np.sqrt(calls2/(n_times-n_fails) - mean_calls**2)
        file.write(f'v{n_v}'+'\t'+f'{mean_calls}'+'\t'+f'{std_calls}'+'\t'+f'{n_fails}'+'\n')
    file.close()


def ADAPT_v_performance() -> None:
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
            vqe = ADAPTVQE(ADAPT_ansatz, method=method)
            vqe.run()
            if vqe.convergence:
                file.write(f'v{n_v}'+'\t'+f'{vqe.tot_operators}'+'\t'+'CONVERGED'+'\n')
            else:
                file.write(f'v{n_v}'+'\t'+f'{vqe.tot_operators}'+'\t'+'FAILED'+'\n')
        file.close()


def ADAPT_table(method: str = 'SLSQP', min_criterion:str = 'Repeated op', tol: float = 1e-10) -> None:
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
                       tol = tol)
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






if __name__ == '__main__':
    #UCC_vs_ADAPT(n_v=1)
    #ADAPT_table(method= 'SLSQP', min_criterion='None',tol=1e-15)
    #ADAPT_v_performance()
    ADAPT_all_v_tracks()