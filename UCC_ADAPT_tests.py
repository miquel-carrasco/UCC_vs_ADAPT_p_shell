from VQE.Nucleus import Nucleus
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz
from VQE.Methods import UCCVQE, OptimizationConvergedException, ADAPTVQE
import numpy as np

def UCC_minimization(nucleus: str,
                     pool_format: str = "Reduced",
                     ref_state: int = 0,
                     opt_method: str = "L-BFGS-B",
                     threshold: float = 1e-6,
                     stop_at_threshold: bool = True):

    nuc = Nucleus(nucleus)
    ref_state = np.eye(nuc.d_H)[:, ref_state]
    ansatz = UCCAnsatz(nucleus = nuc,
                       ref_state = ref_state,
                       pool_format = pool_format)
    vqe = UCCVQE(ansatz = ansatz,
                 method = opt_method,
                 test_threshold = threshold,
                 stop_at_threshold = stop_at_threshold)
    vqe.run()

    return None


def ADAPT_minimization(nucleus: str,
                       pool_format: str = "Reduced",
                       ref_state: int = 0,
                       opt_method: str = "L-BFGS-B",
                       threshold: float = 1e-6,
                       stop_at_threshold: bool = True,
                       max_layers: int = 60,
                       return_data: bool = False):

    
    nuc = Nucleus(nucleus)
    ref_state = np.eye(nuc.d_H)[ref_state]
    ansatz = ADAPTAnsatz(nucleus = nuc,
                       ref_state = ref_state,
                       pool_format = pool_format)
    
    vqe = ADAPTVQE(ansatz = ansatz,
                   method = opt_method,
                   test_threshold = threshold,
                   stop_at_threshold = stop_at_threshold,
                   max_layers = max_layers,
                   return_data = return_data)
    ops = [op.ijkl for op in ansatz.operator_pool]
    if [1, 11, 2, 10] in ops:
        print("Found")    
    if return_data:
        data = vqe.run()
        return data
    else:
        vqe.run()
        return None

if __name__=="__main__":
    UCC_minimization("Li6", ref_state = 4)
    ADAPT_minimization("Li6", ref_state = 4)