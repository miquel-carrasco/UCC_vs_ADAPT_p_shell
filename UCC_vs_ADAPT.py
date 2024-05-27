import numpy as np
from scipy import optimize
from tqdm import tqdm
from time import perf_counter
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import pandas as pd
import concurrent.futures

from VQE.Nucleus import Nucleus
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz
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


def UCC_vs_ADAPT(nuc: str, n_v: int, method: str = 'SLSQP', max_layers: int = 100) -> None:
    nucleus = Nucleus(nuc, 1)
    ref_state = np.eye(nucleus.d_H)[n_v]
    
    ADAPT_ansatz = ADAPTAnsatz(nucleus, ref_state, pool_format='Reduced')
    ADAPT_vqe = ADAPTVQE(ADAPT_ansatz, method = method, max_layers=max_layers)
    ADAPT_vqe.run()
    adapt_rel_error = ADAPT_vqe.rel_error
    adapt_energy = ADAPT_vqe.energy
    adapt_fcalls = ADAPT_vqe.fcalls

    print('ADAPT DONE')

    UCC_ansatz = UCCAnsatz(nucleus, ref_state, pool_format= 'Reduced')
    init_param = np.zeros(len(UCC_ansatz.operator_pool))
    UCC_vqe = UCCVQE(UCC_ansatz,init_param=init_param, method=method, max_layers=max_layers)
    UCC_vqe.run()
    ucc_rel_error = UCC_vqe.rel_error
    ucc_fcalls = UCC_vqe.fcalls

    plt.plot(adapt_fcalls, adapt_rel_error, label='ADAPT-VQE')
    plt.plot(ucc_fcalls, ucc_rel_error, label='UCC-VQE')
    plt.vlines(ADAPT_vqe.layer_fcalls, colors='lightgrey',ymin=1e-8,ymax=10, linestyles='dashed')
    plt.ylim((min([adapt_rel_error[-1],ucc_rel_error[-1]])*0.1), 10)
    plt.xlim(0, (max([ucc_fcalls[-1],adapt_fcalls[-1]])+30))
    plt.yscale('log')
    plt.xlabel('Function calls')
    plt.ylabel('Relative error')
    plt.legend()
    plt.title(f'UCC vs ADAPT-VQE (v{n_v})')

    plt.show()

def v_run(nucleus, ref_state,pool_format, method, test_threshold, stop_at_threshold):
    calls = 0
    fail = False
    ansatz = UCCAnsatz(nucleus=nucleus,ref_state=ref_state,pool_format=pool_format)
    init_params = np.random.uniform(low=-np.pi, high=np.pi, size=len(ansatz.operator_pool))
    random.shuffle(ansatz.operator_pool)
    vqe = UCCVQE(ansatz, init_param=init_params, method=method,
                 test_threshold=test_threshold, stop_at_threshold=stop_at_threshold)
    vqe.run()
    if vqe.success:
        calls += vqe.fcalls[-1]
        fail = False
    else:
        fail = True
    return calls, fail





def UCC_v_performance(nuc: str,
                      method: str,
                      n_vecs: int,
                      n_times: int = 1000,
                      ftol: float = 1e-7,
                      gtol: float = 1e-3,
                      rhoend: float = 1e-5,
                      test_threshold: float = 1e-6,
                      stop_at_threshold: bool = True,
                      pool_format: str = 'Only acting') -> None:
    nucleus = Nucleus(nuc, 1)
    vecs = np.eye(nucleus.d_H)

    try:
        os.makedirs(f'outputs/{nuc}/v_performance/UCC_{pool_format}')
    except OSError:
        pass
    output_folder = os.path.join(f'outputs/{nuc}/v_performance/UCC_{pool_format}')


    file = open(os.path.join(output_folder, f'{method}_performance_randomt0_ntimes={n_times}_pool={pool_format}.dat'), 'w')
    for n_v in range(n_vecs):
        print(f'{n_v}')

        calls = 0
        calls2 = 0
        n_fails = 0
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in range(n_times):
                future = executor.submit(v_run, nucleus, vecs[n_v], pool_format, method, test_threshold, stop_at_threshold)
                futures.append(future)
            for future in tqdm(concurrent.futures.as_completed(futures)):
                callsi, fail = future.result()
                if fail==False:
                    calls += callsi
                    calls2 += callsi**2
                else:
                    n_fails += 1
        if (n_times-n_fails)==0:
            mean_calls = 0
            std_calls = 0
        else:
            mean_calls = calls/(n_times-n_fails)
            std_calls = np.sqrt(calls2/(n_times-n_fails) - mean_calls**2)
        file.write(f'v{n_v}'+'\t'+f'{mean_calls}'+'\t'+f'{std_calls}'+'\t'+f'{n_fails}'+'\n')


    print('Random')
    calls = 0
    calls2 = 0
    n_fails = 0
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in range(n_times):
            ref_state = np.random.rand(nucleus.d_H)
            ref_state = ref_state/np.linalg.norm(ref_state)
            future = executor.submit(v_run, nucleus, ref_state, pool_format, method, test_threshold, stop_at_threshold)
            futures.append(future)
        for future in tqdm(concurrent.futures.as_completed(futures)):
            callsi, fail = future.result()
            if fail==False:
                calls += callsi
                calls2 += callsi**2
            else:
                n_fails += 1
    if (n_times-n_fails)==0:
        mean_calls = 0
        std_calls = 0
    else:
        mean_calls = calls/(n_times-n_fails)
        std_calls = np.sqrt(calls2/(n_times-n_fails) - mean_calls**2)
    file.write(f'random'+'\t'+f'{mean_calls}'+'\t'+f'{std_calls}'+'\t'+f'{n_fails}'+'\n')

    file.close()




if __name__ == '__main__':
    # time1 = perf_counter()
    # Li6 = Nucleus('Li6', 1)
    # ref_state = np.eye(Li6.d_H)[0]
    # UCC_ansatz = UCCAnsatz(Li6, ref_state = ref_state, pool_format = 'Only acting')
    # UCC_vqe = UCCVQE(UCC_ansatz, method = 'SLSQP', test_threshold=1e-6, stop_at_threshold=True)
    # UCC_vqe.run()
    # print(UCC_vqe.fcalls[-1])
    # time2 = perf_counter()
    # print(f'Elapsed time: {time2-time1}')

    # time1 = perf_counter()
    # He8 = Nucleus('He8', 1)
    # ref_state = np.eye(He8.d_H)[0]
    # ADAPT_ansatz = ADAPTAnsatz(He8, ref_state = ref_state, pool_format = 'Reduced')
    # ADAPT_vqe = ADAPTVQE(ADAPT_ansatz, method = 'L-BFGS-B', test_threshold=1e-4, stop_at_threshold=True, max_layers=60)
    # ADAPT_vqe.run_one_step()
    # time2 = perf_counter()
    # print(f'Elapsed time: {time2-time1}')

    UCC_v_performance(nuc='He8', method='SLSQP', n_vecs=5, n_times=50, pool_format='Reduced', test_threshold=1e-4, stop_at_threshold=True)