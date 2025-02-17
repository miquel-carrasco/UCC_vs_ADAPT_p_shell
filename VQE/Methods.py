from scipy.optimize import minimize
import numpy as np
from VQE.Ansatze import UCCAnsatz, ADAPTAnsatz
import scipy
from scipy.sparse import csc_matrix

class OptimizationConvergedException(Exception):
    pass


class VQE():
    """
    Parent class to define the Variational Quantum Eigensolvers (VQEs).

    Attributes:
        method (str): Optimization method.
        test_threshold (float): Threshold to stop the optimization.
        stop_at_threshold (bool): If True, the optimization stops when the threshold is reached.
        fcalls (list): List of function calls.
        energy (list): List of energies.
        rel_error (list): List of relative errors.
        success (bool): If True, the optimization was successful.
        tot_operations (list): List of total operations.
        options (dict): Optimization options.
    
    Methods:
        update_options: Update the optimization options.
    """

    def __init__(self,
                 test_threshold: float = 1e-4,
                 method: str = 'L-BFGS-B',
                 ftol: float = 1e-7,
                 gtol: float = 1e-3,
                 rhoend: float = 1e-5,
                 stop_at_threshold: bool = True) -> None:
        """
        Initialization of the VQE object.

        Args:
            test_threshold (float): Threshold to stop the optimization.
            method (str): Optimization method.
            ftol (float): Tolerance for the energy.
            gtol (float): Tolerance for the gradient.
            rhoend (float): Tolerance for the constraints.
            stop_at_threshold (bool): If True, the optimization stops when the threshold is reached.
        """
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
            print('Invalid optimization method, try: SLSQP, COBYLA, L-BFGS-B or BFGS')
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
    """
    Child class to define the Unitary Coupled Cluster (UCC) VQE.

    Attributes:
        ansatz (UCCAnsatz): UCC Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        final_parameters (list): Final list of parameters.

    Methods:
        run: Runs the VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
    """

    def __init__(self, 
                 ansatz: UCCAnsatz,
                 init_param: list = [],
                 test_threshold: float = 1e-4,
                 method: str = 'L-BFGS-B',
                 stop_at_threshold: bool = True) -> None:
        """
        Initialization of the UCCVQE object.

        Args:
            Ansatz (UCCAnsatz): UCC Ansatz object.
            init_param (list): Initial list of parameters.
            test_threshold (float): Threshold to stop the optimization.
            method (str): Optimization method.
            stop_at_threshold (bool): If True, the optimization stops when the threshold is reached.
        """
        super().__init__(test_threshold=test_threshold, method=method, stop_at_threshold=stop_at_threshold)
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        if len(init_param) == 0:
            self.parameters = np.random.uniform(-np.pi,np.pi,(len(self.ansatz.operator_pool)))
        else:
            self.parameters = init_param

    
    def run(self) -> float:
        """
        Runs the VQE algorithm
        
        Returns:
            float: Final minimized energy of the system.
        """

        # print("\n\n")
        # print(" --------------------------------------------------------------------------")
        # print("                              UCC for ", self.nucleus.name)                 
        # print(" --------------------------------------------------------------------------")
        # print('\n\n')

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
        # print(f'The UCC converged after {self.tot_operations[-1]} operations')
        # print(f'The final relative error is {self.rel_error[-1]}')
    

    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.

        Args:
            params (list): List of parameters.
        """

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


class ADAPTVQE(VQE):
    """
    Child class to define the ADAPT VQE.

    Attributes:
        ansatz (ADAPTAnsatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        tot_operators (int): Total number of operators.
        layer_fcalls (list): List of function calls per layer.
        state_layers (list): List of states per layer.
        parameter_layers (list): List of parameters per layer.
        max_layers (int): Maximum number of layers.
        return_data (bool): If True, returns the data of the optimization.
    
    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """
    def __init__(self, 
                 ansatz: ADAPTAnsatz,
                 method: str = 'L-BFGS-B',
                 conv_criterion: str = 'Repeated op',
                 test_threshold: float = 1e-4,
                 stop_at_threshold: bool = True,
                 max_layers: int = 100,
                 return_data: bool = False) -> None:
        
        super().__init__(test_threshold = test_threshold, method = method, stop_at_threshold = stop_at_threshold)
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []
        self.tot_operators = 0
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
    
    def run(self) -> tuple:
        """
        Runs the ADAPT VQE algorithm and returns the data of the optimization, if return_data is True.

        Returns:
            list: List of the selected operator per layer.
            list: List of energy gradient after optimization per layer.
            list: List of energies per layer.
            list: List of relative errors per layer.
            list: List of function calls per layer.        
        """
        print("\n\n")
        print(" --------------------------------------------------------------------------")
        print("                            ADAPT for ", self.nucleus.name)                 
        print(" --------------------------------------------------------------------------")
        print('\n\n')

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
            self.ansatz.added_operators.append(next_operator)
            gradient_layers.append(next_gradient)
            self.parameter_layers.append([])
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
                next_operator,next_gradient = self.ansatz.choose_operator()
                if self.conv_criterion == 'Repeated op' and next_operator == self.ansatz.added_operators[-1]:
                    self.ansatz.minimum = True
                elif self.conv_criterion == 'Gradient' and next_gradient < 1e-7:
                    self.ansatz.minimum = True
                else:
                    energy_layers.append(self.energy[-1])
                    rel_error_layers.append(self.rel_error[-1])
                    fcalls_layers.append(self.fcalls[-1])
                print('\n')
                print(f"------------ LAYER {len(energy_layers)-1} ------------")
                print('Energy: ',energy_layers[-1])
                print('Rel. Error: ',rel_error_layers[-1])
                print('New Operator: ',self.ansatz.added_operators[-1].ijkl)
                print(csc_matrix(self.ansatz.added_operators[-1].matrix))
            except OptimizationConvergedException:
                if self.return_data:
                    opt_grad_layers.append('Manually stopped')
            self.state_layers.append(self.ansatz.ansatz)
            for a in range(len(self.parameters)):
                self.parameter_layers[a].append(self.parameters[a])      
            rel_error = abs((self.energy[-1] - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
            if rel_error < self.test_threshold and self.stop_at_threshold:
                self.success = True

                self.ansatz.minimum = True
                break
        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])
        fcalls_layers.append(self.fcalls[-1])
        print('\n')
        print(f"------------ LAYER {len(energy_layers)-1} ------------")
        print('Energy: ',energy_layers[-1])
        print('Rel. Error: ',rel_error_layers[-1])
        print('New operator: ',self.ansatz.added_operators[-1].ijkl)
        print('\n')
        print(f'The ADAPT converged after {self.tot_operations[-1]} operations')
        print(f'The final relative error is {self.rel_error[-1]}')
        if self.conv_criterion == 'None' and self.ansatz.minimum == False:
            self.ansatz.minimum = True
            opt_grad_layers.append('Manually stopped')
        if self.return_data:
            return  gradient_layers, opt_grad_layers, energy_layers, rel_error_layers, fcalls_layers
        

    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """
        self.ansatz.count_fcalls = False
        E = self.ansatz.energy(params)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=(self.fcalls[-1]-self.fcalls[-2])*len(self.ansatz.added_operators)
        self.tot_operations.append(self.tot_operators)
        if self.rel_error[-1] < self.test_threshold and self.stop_at_threshold:
            self.success = True
            self.ansatz.minimum = True
            self.parameters = params
            raise OptimizationConvergedException
