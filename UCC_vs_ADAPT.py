import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from time import perf_counter
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm


from VQE.Nucleus import Nucleus
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz
from VQE.Utils import labels_all_combinations
from VQE.Methods import UCCVQE, OptimizationConvergedException, ADAPTVQE


def UCC_vs_adapt():
    Li6 = Nucleus('Li6', 1)
    ref_state = np.eye(Li6.d_H)[0]
    
    ADAPT_ansatz = ADAPTAnsatz(Li6, ref_state)
    ADAPT_vqe = ADAPTVQE(ADAPT_ansatz, method='SLSQP')
    ADAPT_vqe.run()
    adapt_rel_error = ADAPT_vqe.rel_error
    adapt_fcalls = ADAPT_vqe.fcalls

    UCC_ansatz = UCCAnsatz(Li6, ref_state)
    init_param = np.zeros(len(UCC_ansatz.operators))
    UCC_vqe = UCCVQE(UCC_ansatz,init_param=init_param, method='SLSQP')
    UCC_vqe.run()
    ucc_rel_error = UCC_vqe.rel_error
    ucc_fcalls = UCC_vqe.fcalls

    plt.plot(adapt_fcalls, adapt_rel_error, label='ADAPT-VQE')
    plt.plot(ucc_fcalls, ucc_rel_error, label='UCC-VQE')
    plt.vlines(ADAPT_vqe.layer_fcalls, colors='lightgrey',ymin=1e-8,ymax=10, linestyles='dashed')
    plt.ylim(1e-2, 10)
    plt.xlim(0, 200)
    plt.yscale('log')
    plt.legend()

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
                file.write(f'v{n_v}'+'\t'+f'{vqe.fcalls[-1]}'+'\t'+'CONVERGED'+'\n')
            else:
                file.write(f'v{n_v}'+'\t'+f'{vqe.fcalls[-1]}'+'\t'+'FAILED'+'\n')
        file.close()




if __name__ == '__main__':
    UCC_v_performance('COBYLA', 1000)