from scipy.optimize import minimize
import numpy as np
import random
import matplotlib.pyplot as plt

from .Nucleus import Nucleus, TwoBodyExcitationOperator
from .Ansatze import UCCAnsatz, ADAPTAnsatz

class OptimizationConvergedException(Exception):
    pass


class VQE():
    """Class to define the Variational Quantum Eigensolver (VQE) algorithm"""

    def __init__(self, test_threshold: float = 1e-6,
               method: str = 'SLSQP',
               ftol: float = 1e-7,
               gtol: float = 1e-3,
               rhoend: float = 1e-5,
               stop_at_threshold: bool = True) -> None:

        self.method = method
        self.test_threshold = test_threshold
        self.stop_at_threshold = stop_at_threshold
        self.fcalls = []
        self.energy = []
        self.rel_error = []
        self.success = False 

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
                 init_param: list,
                 test_threshold: float = 1e-6,
                 method: str = 'SLSQP',
                 ftol: float = 1e-7,
                 gtol: float = 1e-3,
                 rhoend: float = 1e-5,
                 stop_at_threshold: bool = True) -> None:
        
        super().__init__(test_threshold=test_threshold, method=method, ftol=ftol,
                         gtol=gtol, rhoend=rhoend, stop_at_threshold=stop_at_threshold)
        
        self.ansatz = Ansatz
        self.nucleus = Ansatz.nucleus
        self.parameters = init_param

    
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
        if self.rel_error[-1] < self.test_threshold:
            self.success = True
            raise OptimizationConvergedException


class ADAPTVQE(VQE):

    def __init__(self, 
                 Ansatz: ADAPTAnsatz,
                 method: str = 'SLSQP',
                 conv_criterion: str = 'Repeated op',
                 test_threshold: float = 1e-6,
                 stop_at_threshold: bool = True,
                 ftol: float = 1e-7,
                 gtol: float = 1e-3,
                 rhoend: float = 1e-5,
                 tol_method: str = 'Manual',
                 max_layers: int = 100,
                 return_data: bool = False) -> None:
        
        super().__init__(test_threshold, method, ftol, gtol, rhoend, stop_at_threshold)
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

        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<=self.max_layers:
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
                elif self.conv_criterion == 'Gradient' and opt_grad < 1e-6:
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

        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])
        fcalls_layers.append(self.fcalls[-1])
        if self.conv_criterion == 'None' and self.ansatz.minimum == False:
            self.ansatz.minimum = True
            opt_grad_layers.append('Manually stopped')
        if self.return_data:
            return  gradient_layers, opt_grad_layers, energy_layers, rel_error_layers, fcalls_layers
        
    
    def run_one_step(self, final_run: bool = True) -> tuple:
        """Runs the ADAPT VQE algorithm one step at a time"""

        self.ansatz.fcalls = 0
        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=self.fcalls[-1]*len(self.ansatz.added_operators)
        self.tot_operators_layers.append(self.tot_operators)
        first_operator,first_gradient = self.ansatz.choose_operator()
        gradient_layers = [first_gradient]
        opt_grad_layers = []
        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]
        fcalls_layers = [self.fcalls[-1]]
        self.state_layers.append(self.ansatz.ansatz)
        self.ansatz.added_operators.append(first_operator)

        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<self.max_layers+1:
            self.parameter_layers.append([])
            if self.tol_method == 'Automatic':    
                ftol = gradient_layers[-1]*1e-6
                gtol = gradient_layers[-1]*1e-2
                rhoend = gradient_layers[-1]*1e-2
                self.update_options(ftol,gtol,rhoend)
            self.layer_fcalls.append(self.ansatz.fcalls)
            self.ansatz.count_fcalls = True
            try:
                result = minimize(self.ansatz.energy_one_step,
                                  0.0,
                                  method=self.method,
                                  callback=self.callback_one_step,
                                  options=self.options)
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
                elif self.conv_criterion == 'Gradient' and opt_grad < 1e-6:
                    self.ansatz.minimum = True
                else:
                    if len(self.ansatz.added_operators)<self.max_layers:
                        self.ansatz.added_operators.append(next_operator)
                        gradient_layers.append(next_gradient)
                        energy_layers.append(self.energy[-1])
                        rel_error_layers.append(self.rel_error[-1])
                        fcalls_layers.append(self.fcalls[-1])
                    else:
                        self.ansatz.minimum = True
            except OptimizationConvergedException:
                if self.return_data:
                    opt_grad_layers.append('Manually stopped')
            self.state_layers.append(self.ansatz.ansatz)
            for a in range(len(self.parameters)):
                self.parameter_layers[a].append(self.parameters[a])
        

        if final_run:
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
                next_operator,next_gradient = self.ansatz.choose_operator()
                if self.conv_criterion == 'Repeated op' and next_operator == self.ansatz.added_operators[-1]:
                    self.ansatz.minimum = True
                elif self.conv_criterion == 'Gradient' and opt_grad < 1e-6:
                    self.ansatz.minimum = True
                else:
                    self.ansatz.added_operators.append(next_operator)
                    gradient_layers.append(next_gradient)
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
        

    def run_n_layers(self, n_layers: int) -> tuple:
        """Runs the ADAPT VQE algorithm for a given number of layers"""

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

        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<=self.max_layers:
            self.parameter_layers.append([])
            gradient_layers.append(next_gradient)
            self.ansatz.added_operators.append(next_operator)
            if self.tol_method == 'Automatic':    
                ftol = gradient_layers[-1]*1e-6
                gtol = gradient_layers[-1]*1e-2
                rhoend = gradient_layers[-1]*1e-2
                self.update_options(ftol,gtol,rhoend)
            self.layer_fcalls.append(self.ansatz.fcalls)
            self.parameters.append(0.0)
            self.ansatz.count_fcalls = True
            try:
                result = minimize(self.ansatz.energy_n_layers,
                                  args=(n_layers),
                                  x0=self.parameters,
                                  method=self.method,
                                  callback=self.callback_n_layers,
                                  options=self.options)
                self.parameters = list(result.x)
                if self.return_data:
                    if self.method!='COBYLA':
                        opt_grad= np.linalg.norm(result.jac)
                    else:
                        opt_grad=0
                    opt_grad_layers.append(opt_grad)
                self.ansatz.count_fcalls = False
                self.ansatz.ansatz = self.ansatz.build_ansatz_n_layers(self.parameters,n_layers)
                next_operator,next_gradient = self.ansatz.choose_operator()
                if self.conv_criterion == 'Repeated op' and next_operator == self.ansatz.added_operators[-1]:
                    self.ansatz.minimum = True
                elif self.conv_criterion == 'Gradient' and opt_grad < 1e-6:
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
        self.tot_operators_layers.append(self.tot_operators)
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
        self.tot_operators_layers.append(self.tot_operators)
        if self.rel_error[-1] < self.test_threshold and self.stop_at_threshold:
            self.success = True
            self.ansatz.minimum = True
            self.parameters = param
            raise OptimizationConvergedException
    
    def callback_n_layers(self, params: float, n_layers) -> None:
        """Callback function to store the energy and parameters at each iteration
        and stop the optimization if the threshold is reached."""

        self.ansatz.count_fcalls = False
        E = self.ansatz.energy_n_layers(params,n_layers)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=(self.fcalls[-1]-self.fcalls[-2])*len(self.ansatz.added_operators)
        self.tot_operators_layers.append(self.tot_operators)
        if self.rel_error[-1] < self.test_threshold and self.stop_at_threshold:
            self.success = True
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
