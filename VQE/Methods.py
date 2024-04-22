from scipy.optimize import minimize
import numpy as np
import random
import matplotlib.pyplot as plt

from .Nucleus import Nucleus
from .Ansatze import UCCAnsatz, ADAPTAnsatz

class OptimizationConvergedException(Exception):
    pass

class UCCVQE():
    """Class to define the Variational Quantum Eigensolver (VQE) algorithm"""

    def __init__(self, Ansatz: UCCAnsatz, init_param: list[float], 
                 precision: float = 1e-6, method: str = 'SLSQP') -> None:
        
        self.ansatz = Ansatz
        self.method = method
        self.parameters = init_param
        self.precision = precision
        self.fcalls = []
        self.energy = []
        self.rel_error = []
        self.final_parameters = []
        self.convergence = False

    
    def run(self) -> float:
        """Runs the VQE algorithm"""

        self.ansatz.fcalls = 0
        E0 = self.ansatz.energy(self.parameters)

        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.ansatz.count_fcalls = True
        try:
            result = minimize(self.ansatz.energy, self.parameters, method=self.method, callback=self.callback)
        except OptimizationConvergedException:
            pass
        self.ansatz.count_fcalls = False
    
    def callback(self, params: list[float]) -> None:
        """Callback function to store the energy and parameters at each iteration
        and stop the optimization if the precision is reached."""

        self.ansatz.count_fcalls = False
        E = self.ansatz.energy(params)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.final_parameters = params
        if self.rel_error[-1] < self.precision:
            self.convergence = True
            raise OptimizationConvergedException


class ADAPTVQE():

    def __init__(self, Ansatz: ADAPTAnsatz, precision: float = 1e-6, method: str = 'SLSQP') -> None:
        
        self.nucleus = Ansatz.nucleus
        self.ansatz = Ansatz
        self.method = method
        self.precision = precision
        self.fcalls = []
        self.energy = []
        self.rel_error = []
        self.parameters = []
        self.convergence = False
        self.layer_fcalls = []
    
    def run(self) -> float:
        """Runs the ADAPT VQE algorithm"""

        self.ansatz.fcalls = 0
        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.ansatz.choose_operator()
        while self.ansatz.minimum == False and self.fcalls[-1] < 700:
            self.layer_fcalls.append(self.ansatz.fcalls)
            print(self.ansatz.fcalls)
            self.parameters.append(0.0)
            self.ansatz.count_fcalls = True
            try:
                result = minimize(self.ansatz.energy, self.parameters, method=self.method, callback=self.callback)
                self.parameters = list(result.x)
            except OptimizationConvergedException:
                pass
            self.ansatz.count_fcalls = False
            self.ansatz.ansatz = self.ansatz.build_ansatz(self.parameters)
            self.ansatz.choose_operator()
        
    def callback(self, params: list[float]) -> None:
        """Callback function to store the energy and parameters at each iteration
        and stop the optimization if the precision is reached."""

        self.ansatz.count_fcalls = False
        E = self.ansatz.energy(params)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        if self.rel_error[-1] < self.precision:
            self.convergence = True
            self.ansatz.minimum = True
            self.parameters = params
            raise OptimizationConvergedException



if __name__ == '__main__':
    Li6 = Nucleus('Li6', 1)
    ref_state = np.eye(Li6.d_H)[1]
    UCC_ansatz = UCCAnsatz(Li6, ref_state)
    vqe = UCCVQE(UCC_ansatz, np.zeros(len(UCC_ansatz.operators)))
    vqe.run()
    t_fin = vqe.final_parameters
    t0 = np.random.rand(len(t_fin))

    t3 = np.linspace(-7,7,1000)
    for n in range(len(t0)):
        E = [UCC_ansatz.lanscape(t_fin,t,n) for t in t3]
        plt.plot(t3,E)
        print(UCC_ansatz.lanscape(t_fin,0,n)-UCC_ansatz.lanscape(t_fin,2*np.pi,n))
    plt.show()
