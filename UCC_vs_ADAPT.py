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
# plt.rcParams.update(params)


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

def v_run_UCC(nucleus, n_times, ref_state,pool_format, method, test_threshold, stop_at_threshold):
    gates = 0
    gates2 = 0
    n_fails = 0
    for n in range(n_times):    
        fail = False
        ansatz = UCCAnsatz(nucleus=nucleus,ref_state=ref_state,pool_format=pool_format)
        init_params = np.random.uniform(low=-np.pi, high=np.pi, size=len(ansatz.operator_pool))
        random.shuffle(ansatz.operator_pool)
        vqe = UCCVQE(ansatz, init_param=init_params, method=method,
                    test_threshold=test_threshold, stop_at_threshold=stop_at_threshold)
        vqe.run()
        if vqe.success:
            gates += vqe.fcalls[-1]*len(vqe.ansatz.operator_pool)
            gates2 += (vqe.fcalls[-1]*len(vqe.ansatz.operator_pool))**2
        else:
            n_fails += 1
    if (n_times-n_fails)==0:
        mean_gates = 0
        std_gates = 0
    else:
        mean_gates = gates/(n_times-n_fails)
        std_gates = np.sqrt(gates2/(n_times-n_fails) - mean_gates**2)
    return mean_gates, std_gates, n_fails

def vrun_1time_UCC(nucleus, ref_state, pool_format, method, test_threshold, stop_at_threshold):
    fail = False
    ansatz = UCCAnsatz(nucleus=nucleus,ref_state=ref_state,pool_format=pool_format)
    init_params = np.random.uniform(low=-np.pi, high=np.pi, size=len(ansatz.operator_pool))
    random.shuffle(ansatz.operator_pool)
    vqe = UCCVQE(ansatz, init_param=init_params, method=method,
                test_threshold=test_threshold, stop_at_threshold=stop_at_threshold)
    start = perf_counter()
    vqe.run()
    end = perf_counter()
    print(f'Elapsed time: {end-start} seconds')
    return vqe.fcalls[-1]*len(vqe.ansatz.operator_pool), vqe.success


def UCC_v_performance(nuc: str,
                      method: str,
                      n_vecs: int,
                      n_times: int = 1000,
                      test_threshold: float = 1e-4,
                      stop_at_threshold: bool = True,
                      pool_format: str = 'Only acting') -> None:
    nucleus = Nucleus(nuc, 1)
    vecs = np.eye(nucleus.d_H)

    try:
        os.makedirs(f'outputs/{nuc}/v_performance/UCC_{pool_format}')
    except OSError:
        pass
    output_folder = os.path.join(f'outputs/{nuc}/v_performance/UCC_{pool_format}')


    file = open(os.path.join(output_folder, f'{method}_ntimes={n_times}.dat'), 'w')
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for n_v in range(n_vecs):
            future = executor.submit(v_run_UCC, nucleus, n_times, vecs[n_v], pool_format, method, test_threshold, stop_at_threshold)
            futures.append(future)
        rand_state = np.random.uniform(low=-1, high=1, size=nucleus.d_H)
        rand_state = rand_state/np.linalg.norm(rand_state)
        future_random = executor.submit(v_run_UCC, nucleus, n_times, rand_state, pool_format, method, test_threshold, stop_at_threshold)
        for n_v,future in tqdm(enumerate(futures)):
            mean_gates, std_gates, n_fails = future.result()
            file.write(f'v{n_v}'+'\t'+f'{mean_gates}'+'\t'+f'{std_gates}'+'\t'+f'{n_fails}'+'\n')
        mean_gates, std_gates, n_fails = future_random.result()
        file.write(f'random'+'\t'+f'{mean_gates}'+'\t'+f'{std_gates}'+'\t'+f'{n_fails}'+'\n')

    file.close()

def UCC_v_performance_2(nuc: str,
                      method: str,
                      n_vecs: int,
                      n_times: int = 1000,
                      test_threshold: float = 1e-4,
                      stop_at_threshold: bool = True,
                      pool_format: str = 'Only acting') -> None:
    nucleus = Nucleus(nuc, 1)
    vecs = np.eye(nucleus.d_H)

    try:
        os.makedirs(f'outputs/{nuc}/v_performance/UCC_{pool_format}')
    except OSError:
        pass
    output_folder = os.path.join(f'outputs/{nuc}/v_performance/UCC_{pool_format}')


    file = open(os.path.join(output_folder, f'{method}_ntimes={n_times}.dat'), 'w')

    times_list = [50 for _ in range (n_times//50)]
    if n_times%50 != 0:
        times_list.append(n_times%50)

    vecs_list = range(n_vecs)
    for n_v in tqdm(vecs_list):
        gates = 0
        gates2 = 0
        n_fails = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for N in times_list:
                futures = []
                for _ in range(N):
                    future = executor.submit(vrun_1time_UCC, nucleus, vecs[n_v], pool_format, method, test_threshold, stop_at_threshold)
                    futures.append(future)
                for future in futures:
                    n_gates, success = future.result()
                    if success:
                        gates += n_gates
                        gates2 += n_gates**2
                    else:
                        n_fails += 1
        if (n_times-n_fails)==0:
            mean_gates = 0
            std_gates = 0
        else:
            mean_gates = gates/(n_times-n_fails)
            std_gates = np.sqrt(gates2/(n_times-n_fails) - mean_gates**2)
        file.write(f'v{n_v}'+'\t'+f'{mean_gates}'+'\t'+f'{std_gates}'+'\t'+f'{n_fails}'+'\n')
    
    rand_state = np.random.uniform(low=-1, high=1, size=nucleus.d_H)
    rand_state = rand_state/np.linalg.norm(rand_state)
    gates = 0
    gates2 = 0
    n_fails = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for N in times_list:
            futures = []
            for _ in range(N):
                future = executor.submit(vrun_1time_UCC, nucleus, rand_state, pool_format, method, test_threshold, stop_at_threshold)
                futures.append(future)
            for future in futures:
                n_gates, success = future.result()
                if success:
                    gates += n_gates
                    gates2 += n_gates**2
                else:
                    n_fails += 1
    if (n_times-n_fails)==0:
        mean_gates = 0
        std_gates = 0
    else:
        mean_gates = gates/(n_times-n_fails)
        std_gates = np.sqrt(gates2/(n_times-n_fails) - mean_gates**2)
    file.write(f'random'+'\t'+f'{mean_gates}'+'\t'+f'{std_gates}'+'\t'+f'{n_fails}'+'\n')

    file.close()


def v_run_ADAPT(nucleus, ref_state,pool_format, method, test_threshold, stop_at_threshold, conv_criterion):
    gates = 0
    gates2 = 0 
    fail = False
    ansatz = ADAPTAnsatz(nucleus=nucleus,ref_state=ref_state,pool_format=pool_format)
    E0 = ansatz.E0
    overlap = (ref_state.conj().T @ nucleus.eig_vec[:,0])**2
    vqe = ADAPTVQE(ansatz, method=method, test_threshold=test_threshold, stop_at_threshold=stop_at_threshold, conv_criterion=conv_criterion)
    vqe.run()
    return vqe.tot_operators, vqe.fcalls[-1], len(vqe.ansatz.added_operators), vqe.success, E0, overlap


def ADAPT_v_performance(nuc: str,
                      method: str,
                      n_vecs: int,
                      conv_criterion: str = 'Repeated op',
                      test_threshold: float = 1e-4,
                      stop_at_threshold: bool = True,
                      pool_format: str = 'Only acting',
                      n_times: int = 50) -> None:
    nucleus = Nucleus(nuc, 1)
    basis = np.eye(nucleus.d_H)

    try:
        os.makedirs(f'outputs/{nuc}/v_performance/ADAPT')
    except OSError:
        pass
    output_folder = os.path.join(f'outputs/{nuc}/v_performance/ADAPT')

    
    vecs = []
    energies = []
    overlaps = []
    gates = []
    layers = []
    fcalls = []
    successes = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for n_v in range(nucleus.d_H):
            future = executor.submit(v_run_ADAPT, nucleus, basis[n_v],pool_format, method, test_threshold, stop_at_threshold, conv_criterion)
            futures.append(future)
        for n_v,future in tqdm(enumerate(futures)):
            n_gates, n_fcalls, n_layers, success, E0, overlap = future.result()
            if success:
                successes.append('SUCCESS')
            else: 
                successes.append('FAIL')
            vecs.append(f'v{n_v}')
            energies.append(E0)
            overlaps.append(overlap)
            gates.append(n_gates)
            fcalls.append(n_fcalls)
            layers.append(n_layers)
        data = {'v': vecs, 'Gates': gates, 'Layers': layers, 'Fcalls': fcalls, 'E0': energies, 'Overlap': overlaps, 'Success': successes}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_folder, f'{method}_basis.csv'), sep='\t', index=False)


        print('Random state')
        times_list = [50 for _ in range (n_times//50)]
        if n_times%50 != 0:
            times_list.append(n_times%50)

        vecs = []
        energies = []
        overlaps = []
        gates = []
        layers = []
        fcalls = []
        successes = []
        for N in times_list:
            futures = []
            for _ in range(N):
                rand_state = np.random.uniform(low=-1, high=1, size=nucleus.d_H)
                rand_state = rand_state/np.linalg.norm(rand_state)
                future_random = executor.submit(v_run_ADAPT, nucleus, rand_state,pool_format, method, test_threshold, stop_at_threshold, conv_criterion)
                futures.append(future_random)
            for n_v,future in tqdm(enumerate(futures)):
                n_gates, n_fcalls, n_layers, success, E0, overlap = future.result()
                if success:
                    successes.append('SUCCESS')
                else:
                    successes.append('FAIL')
                energies.append(E0)
                overlaps.append(overlap)
                gates.append(n_gates)
                fcalls.append(n_fcalls)
                layers.append(n_layers)
            data = {'Gates': gates, 'Layers': layers, 'Fcalls': fcalls, 'E0': energies, 'Overlap': overlaps, 'Success': successes}
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(output_folder, f'{method}_random.csv'), sep='\t', index=False)
                



def v_run_seq_ADAPT(nucleus, ref_state,pool_format, method, test_threshold, stop_at_threshold, conv_criterion, max_layers):
    gates = 0
    gates2 = 0 
    fail = False
    ansatz = ADAPTAnsatz(nucleus=nucleus,ref_state=ref_state,pool_format=pool_format)
    vqe = ADAPTVQE(ansatz,
                   method=method,
                   test_threshold=test_threshold,
                   stop_at_threshold=stop_at_threshold,
                   conv_criterion=conv_criterion,
                   max_layers=max_layers)
    
    vqe.run_one_step(final_run=False)
    return vqe.tot_operators, len(vqe.ansatz.added_operators), vqe.success,


def seq_ADAPT_v_performance(nuc: str,
                      method: str,
                      n_vecs: int,
                      conv_criterion: str = 'Repeated op',
                      test_threshold: float = 1e-4,
                      stop_at_threshold: bool = True,
                      pool_format: str = 'Reduced',
                      max_layers: int = 500) -> None:
    nucleus = Nucleus(nuc, 1)
    vecs = np.eye(nucleus.d_H)

    try:
        os.makedirs(f'outputs/{nuc}/v_performance/Seq-ADAPT')
    except OSError:
        pass
    output_folder = os.path.join(f'outputs/{nuc}/v_performance/Seq-ADAPT')


    file = open(os.path.join(output_folder, f'{method}.dat'), 'w')
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for n_v in range(nucleus.d_H):
            print('n_v')
            future = executor.submit(v_run_seq_ADAPT, nucleus, vecs[n_v],pool_format, method, test_threshold, stop_at_threshold, conv_criterion, max_layers)
            futures.append(future)
        for n_v,future in tqdm(enumerate(futures)):
            gates, layers, success, E0, overlap = future.result()
            if success:
                file.write(f'v{n_v}'+'\t'+f'{gates}'+'\t'+f'{layers}'+'\t'+'SUCCESS'+f'E0={E0}'+'\t'+f'overlap={overlap}'+'\n')
            else:
                file.write(f'v{n_v}'+'\t'+f'{gates}'+'\t'+f'{layers}'+'\t'+'FAIL'+f'E0={E0}'+'\t'+f'overlap={overlap}'+'\n')
        
        print('Random state')
        gates = 0
        gates2 = 0
        layers = 0
        layers2 = 0
        n_fails = 0
        futures = []
        for N in range(50):
            rand_state = np.random.uniform(low=-1, high=1, size=nucleus.d_H)
            rand_state = rand_state/np.linalg.norm(rand_state)
            future_random = executor.submit(v_run_seq_ADAPT, nucleus, rand_state,pool_format, method, test_threshold, stop_at_threshold, conv_criterion, max_layers)
            futures.append(future_random)
        for future in tqdm(futures):
            n_gates, n_layers, success = future.result()
            if success:
                gates += n_gates
                gates2 += n_gates**2
                layers += n_layers
                layers2 += n_layers**2
            else:
                n_fails += 1
        if (50-n_fails)==0:
            mean_gates = 0
            std_gates = 0
            mean_layers = 0
        else:
            mean_gates = gates/(50-n_fails)
            std_gates = np.sqrt(gates2/(50-n_fails) - mean_gates**2)
            mean_layers = layers/(50-n_fails)
            std_layers = np.sqrt(layers2/(50-n_fails) - mean_layers**2)

        file.write(f'random'+'\t'+f'{mean_gates}'+'\t'+f'{std_gates}'+'\t'+f'{mean_layers}'+'\t'+f'{std_layers}'+'\t'+f'{n_fails}'+'\n')

    file.close()





if __name__ == '__main__':
#    ADAPT_v_performance('Li6', 'L-BFGS-B',10, conv_criterion='Repeated op', test_threshold=1e-4, stop_at_threshold=True, pool_format='Reduced', n_times=100)
    UCC_v_performance('Li6', 'L-BFGS-B',10, n_times=100, test_threshold=1e-4, stop_at_threshold=True, pool_format='Reduced')