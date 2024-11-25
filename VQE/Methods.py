from scipy.optimize import minimize
import numpy as np
import random
import matplotlib.pyplot as plt

from VQE.Nucleus import Nucleus, TwoBodyExcitationOperator
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz
from time import perf_counter
import scipy

class OptimizationConvergedException(Exception):
    pass


class VQE():
    """Class to define the Variational Quantum Eigensolver (VQE) algorithm"""

    def __init__(self, test_threshold: float = 1e-4,
               method: str = 'SLSQP',
               ftol: float = 1e-7,
               gtol: float = 1e-5,
               rhoend: float = 1e-5,
               stop_at_threshold: bool = True) -> None:

        self.method = method
        self.test_threshold = test_threshold
        self.stop_at_threshold = stop_at_threshold
        self.fcalls = []
        self.energy = []
        self.rel_error = []
        self.success = False 
        self.tot_operations = [0]

        try:
            self.method = method
        except method not in ['SLSQP', 'COBYLA','L-BFGS-B','BFGS']:
            print('Invalid optimization method')
            exit()
        
        self.options={}
        if self.method in ['SLSQP','L-BFGS-B']:
            self.options.setdefault('ftol',ftol)
        if self.method in ['L-BFGS-B','BFGS']:
            self.options.setdefault('gtol',gtol)
        if self.method == 'COBYLA':
            self.options.setdefault('tol',rhoend)

    def update_options(self,ftol,gtol,rhoend) -> None:
        """Update the optimization options"""

        if self.method in ['SLSQP','L-BFGS-B']:
            self.options['ftol']=ftol
        if self.method in ['L-BFGS-B','BFGS']:
            self.options['gtol']=gtol
        if self.method == 'COBYLA':
            self.options['rhoend']=rhoend


class UCCVQE(VQE):

    def __init__(self, Ansatz: UCCAnsatz,
                 init_param: list = [],
                 test_threshold: float = 1e-4,
                 method: str = 'SLSQP',
                 stop_at_threshold: bool = True) -> None:
        
        super().__init__(test_threshold=test_threshold, method=method, stop_at_threshold=stop_at_threshold)
        
        self.ansatz = Ansatz
        self.nucleus = Ansatz.nucleus
        if len(init_param) == 0:
            self.parameters = np.random.uniform(-np.pi,np.pi,(len(self.ansatz.operator_pool)))
        else:
            self.parameters = init_param

    
    def run(self) -> float:
        """Runs the VQE algorithm"""

        # print("\n\n\n")
        # print(" --------------------------------------------------------------------------")
        # print("                              UCC for ", self.nucleus.name)                 
        # print(" --------------------------------------------------------------------------")

        self.ansatz.fcalls = 0
        self.ansatz.count_fcalls = False
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
    
    def sequential_run(self, final_run: bool = True) -> float:
        """Runs the VQE algorithm"""

        # print("\n\n\n")
        # print(" --------------------------------------------------------------------------")
        # print("                            Seq-UCC for ", self.nucleus.name)                 
        # print(" --------------------------------------------------------------------------")

        self.ansatz.fcalls = 0
        self.ansatz.n_layers = 0
        E0 = self.ansatz.sequential_energy(parameter=[0.0])
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.ansatz.count_fcalls = True

        for n_layer in range(1,len(self.ansatz.operator_pool)+1):
            print(self.ansatz.ansatz)
            self.ansatz.n_layers = n_layer
            parameter=np.zeros(1)
            try:
                result = minimize(self.ansatz.sequential_energy, list(parameter), method=self.method, callback=self.sequential_callback, bounds=[(-np.pi,np.pi)])
            except OptimizationConvergedException:
                pass
            self.ansatz.count_fcalls = False


    def callback(self, params: list) -> None:
        """Callback function to store the energy and parameters at each iteration
        and stop the optimization if the threshold is reached."""

        self.ansatz.count_fcalls = False
        E = self.ansatz.energy(params)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.final_parameters = params
        # print(f' Energy: {E}')
        self.tot_operations.append(self.fcalls[-1]*len(self.ansatz.operator_pool))
        if self.rel_error[-1] < self.test_threshold:
            self.success = True
            raise OptimizationConvergedException
    
    def sequential_callback(self, param: float) -> None:

        self.ansatz.count_fcalls = False
        E = self.ansatz.sequential_energy(param)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.parameters[self.ansatz.n_layers-1] = param
        # print(f' Energy: {E}')
        if self.rel_error[-1] < self.test_threshold:
            self.success = True
            raise OptimizationConvergedException


class ADAPTVQE(VQE):

    def __init__(self, 
                 Ansatz: ADAPTAnsatz,
                 method: str = 'SLSQP',
                 conv_criterion: str = 'Repeated op',
                 test_threshold: float = 1e-4,
                 stop_at_threshold: bool = True,
                 tol_method: str = 'Manual',
                 max_layers: int = 100,
                 return_data: bool = False) -> None:
        
        super().__init__(test_threshold = test_threshold, method = method, stop_at_threshold = stop_at_threshold)
        self.ansatz = Ansatz
        self.nucleus = Ansatz.nucleus
        self.parameters = []
        self.tot_operators=0
        self.tot_operators_layers=[]
        self.layer_fcalls = []
        self.state_layers = []
        self.parameter_layers = []
        self.max_layers = max_layers
        self.return_data = return_data

        try:
            self.conv_criterion = conv_criterion
        except conv_criterion not in ['Repeated op', 'Gradient','None']:
            print('Invalid minimum criterion. Choose between "Repeated op", "Gradient" and "None"')
            exit()

        try:
            self.tol_method = tol_method
        except tol_method not in ['Manual', 'Automatic']:
            print('Invalid tolerance method. Choose between "Manual" and "Automatic"')
            exit()
    
    def run(self) -> tuple:
        """Runs the ADAPT VQE algorithm"""

        # print("\n\n\n")
        # print(" --------------------------------------------------------------------------")
        # print("                            ADAPT for ", self.nucleus.name)                 
        # print(" --------------------------------------------------------------------------")

        self.ansatz.fcalls = 0
        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=self.fcalls[-1]*len(self.ansatz.added_operators)
        print('Initial Energy: ',E0)
        next_operator,next_gradient = self.ansatz.choose_operator()
        gradient_layers = []
        opt_grad_layers = []
        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]
        fcalls_layers = [self.fcalls[-1]]
        self.state_layers.append(self.ansatz.ansatz)

        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<self.max_layers:
            self.tot_operators_layers.append(self.tot_operators)
            self.ansatz.added_operators.append(next_operator)
            gradient_layers.append(next_gradient)
            self.parameter_layers.append([])
            if self.tol_method == 'Automatic':    
                ftol = gradient_layers[-1]*1e-6
                gtol = gradient_layers[-1]*1e-2
                rhoend = gradient_layers[-1]*1e-2
                self.update_options(ftol,gtol,rhoend)
            self.layer_fcalls.append(self.ansatz.fcalls)
            self.parameters.append(0.0)
            self.ansatz.count_fcalls = True
            try:
                result = minimize(self.ansatz.energy,
                                  self.parameters,
                                  method=self.method,
                                  callback=self.callback,
                                  options=self.options)
                self.parameters = list(result.x)
                nf = result.nfev
                if self.return_data:
                    if self.method!='COBYLA':
                        opt_grad= np.linalg.norm(result.jac)
                    else:
                        opt_grad=0
                    opt_grad_layers.append(opt_grad)
                self.ansatz.count_fcalls = False
                self.ansatz.ansatz = self.ansatz.build_ansatz(self.parameters)
                # print("Tot operations: ", self.tot_operators)
                # print("Fcalls: ", nf)
                next_operator,next_gradient = self.ansatz.choose_operator()
                if self.conv_criterion == 'Repeated op' and next_operator == self.ansatz.added_operators[-1]:
                    self.ansatz.minimum = True
                elif self.conv_criterion == 'Gradient' and next_gradient < 1e-7:
                    self.ansatz.minimum = True
                else:
                    energy_layers.append(self.energy[-1])
                    rel_error_layers.append(self.rel_error[-1])
                    fcalls_layers.append(self.fcalls[-1])
                print("LAYER ",len(energy_layers)-2)
                print('Energy: ',energy_layers[-1])
                print('Operators: ',self.ansatz.added_operators[-1].ijkl)
                print(scipy.sparse.coo_matrix(self.ansatz.added_operators[-1].matrix))
                print('Gradient: ',gradient_layers[-1])
                print("Parameters: ",self.parameters)
            except OptimizationConvergedException:
                if self.return_data:
                    opt_grad_layers.append('Manually stopped')
            self.state_layers.append(self.ansatz.ansatz)
            for a in range(len(self.parameters)):
                self.parameter_layers[a].append(self.parameters[a])
            
            rel_error = abs((self.energy[-1] - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
            print(rel_error)
            if rel_error < self.test_threshold and self.stop_at_threshold:
                self.success = True
                self.ansatz.minimum = True
                break

        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])
        fcalls_layers.append(self.fcalls[-1])
        print("LAYER ",len(energy_layers)-2)
        print('Energy: ',energy_layers[-1])
        print('Operators: ',self.ansatz.added_operators[-1].ijkl)
        print('Gradient: ',gradient_layers[-1])
        print("Parameters: ",self.parameters)
        if self.conv_criterion == 'None' and self.ansatz.minimum == False:
            self.ansatz.minimum = True
            opt_grad_layers.append('Manually stopped')
        if self.return_data:
            return  gradient_layers, opt_grad_layers, energy_layers, rel_error_layers, fcalls_layers
        
    
    def run_one_step(self, final_run: bool = True) -> tuple:
        """Runs the ADAPT VQE algorithm"""

        # print("\n\n\n")
        # print(" --------------------------------------------------------------------------")
        # print("                            ADAPT for ", self.nucleus.name)                 
        # print(" --------------------------------------------------------------------------")

        self.ansatz.fcalls = 0
        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=self.fcalls[-1]*len(self.ansatz.added_operators)
        self.tot_operators_layers.append(self.tot_operators)
        next_operator,next_gradient = self.ansatz.choose_operator()
        gradient_layers = []
        opt_grad_layers = []
        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]
        fcalls_layers = [self.fcalls[-1]]
        self.state_layers.append(self.ansatz.ansatz)

        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<self.max_layers:
            self.ansatz.added_operators.append(next_operator)
            gradient_layers.append(next_gradient)
            self.parameter_layers.append([])
            self.layer_fcalls.append(self.ansatz.fcalls)
            self.ansatz.count_fcalls = True
            try:
                result = minimize(self.ansatz.energy_one_step,
                                  0.0,
                                  method=self.method,
                                  callback=self.callback_one_step,
                                  options=self.options,
                                  bounds = [(-np.pi,np.pi)])
                self.parameters.append(float(result.x))
                if self.return_data:
                    if self.method!='COBYLA':
                        opt_grad= np.linalg.norm(result.jac)
                    else:
                        opt_grad=0
                    opt_grad_layers.append(opt_grad)
                self.ansatz.count_fcalls = False
                self.ansatz.ansatz = self.ansatz.build_ansatz(self.parameters)
                next_operator,next_gradient = self.ansatz.choose_operator()
                if self.conv_criterion == 'Repeated op' and next_operator == self.ansatz.added_operators[-1]:
                    self.ansatz.minimum = True
                elif self.conv_criterion == 'Gradient' and next_gradient < 1e-7:
                    self.ansatz.minimum = True
                else:
                    energy_layers.append(self.energy[-1])
                    rel_error_layers.append(self.rel_error[-1])
                    fcalls_layers.append(self.fcalls[-1])
            except OptimizationConvergedException:
                if self.return_data:
                    opt_grad_layers.append('Manually stopped')
            self.state_layers.append(self.ansatz.ansatz)
            for a in range(len(self.parameters)):
                self.parameter_layers[a].append(self.parameters[a])

        
        if final_run and self.ansatz.minimum == False:
            # print('Final run')
            try:
                result = minimize(self.ansatz.energy,
                                  self.parameters,
                                  method=self.method,
                                  callback=self.callback,
                                  options=self.options)
                self.parameters = list(result.x)
                if self.return_data:
                    if self.method!='COBYLA':
                        opt_grad= np.linalg.norm(result.jac)
                    else:
                        opt_grad=0
                    opt_grad_layers.append(opt_grad)
                self.ansatz.count_fcalls = False
                self.ansatz.ansatz = self.ansatz.build_ansatz(self.parameters)
                energy_layers.append(self.energy[-1])
                rel_error_layers.append(self.rel_error[-1])
                fcalls_layers.append(self.fcalls[-1])
            except OptimizationConvergedException:
                if self.return_data:
                    opt_grad_layers.append('Manually stopped')
            self.state_layers.append(self.ansatz.ansatz)
            for a in range(len(self.parameters)):
                self.parameter_layers[a].append(self.parameters[a])

        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])
        fcalls_layers.append(self.fcalls[-1])
        if self.conv_criterion == 'None' and self.ansatz.minimum == False:
            self.ansatz.minimum = True
            opt_grad_layers.append('Manually stopped')


        if self.return_data:
            return  gradient_layers, opt_grad_layers, energy_layers, rel_error_layers, fcalls_layers
        
        


    def callback(self, params: list) -> None:
        """Callback function to store the energy and parameters at each iteration
        and stop the optimization if the threshold is reached."""

        self.ansatz.count_fcalls = False
        E = self.ansatz.energy(params)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=(self.fcalls[-1]-self.fcalls[-2])*len(self.ansatz.added_operators)
        # print(f' Energy: {E}')
        self.tot_operations.append(self.tot_operators)
        if self.rel_error[-1] < self.test_threshold and self.stop_at_threshold:
            self.success = True
            self.ansatz.minimum = True
            self.parameters = params
            raise OptimizationConvergedException


    def callback_one_step(self, param: float) -> None:
        """Callback function to store the energy and parameters at each iteration
        and stop the optimization if the threshold is reached."""

        self.ansatz.count_fcalls = False
        E = self.ansatz.energy_one_step(param)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=(self.fcalls[-1]-self.fcalls[-2])*len(self.ansatz.added_operators)
        # print(f' Energy: {E}')
        self.tot_operators_layers.append(self.tot_operators)
        if self.rel_error[-1] < self.test_threshold and self.stop_at_threshold:
            self.success = True
            self.ansatz.minimum = True
            self.parameters.append(param)
            raise OptimizationConvergedException
    


if __name__ == '__main__':
    nuc = Nucleus('Be8', 1)
    ref_state = np.eye(len(nuc.H))[6]
    ansatz = ADAPTAnsatz(nuc, ref_state, pool_format='Reduced')
    
    ADAPT = ADAPTVQE(ansatz, method='BFGS', conv_criterion='Repeated op', max_layers=100, return_data=True, test_threshold=1e-6, tol_method='Manual')

    gradient_layers, opt_grad_layers, energy_layers, rel_error_layers, fcalls_layers = ADAPT.run()
    print(len(ansatz.added_operators))