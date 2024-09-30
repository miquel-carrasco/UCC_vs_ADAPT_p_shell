import numpy as np
from .Nucleus import Nucleus, TwoBodyExcitationOperator
from scipy.linalg import expm
import random
from numba import jit, cuda
from  scipy.sparse.linalg import expm_multiply
from scipy.sparse import lil_matrix, csc_matrix
from time import perf_counter
import threading


class Ansatz():
    """Class to define ansätze"""

    def __init__(self,
                 nucleus: Nucleus,
                 ref_state: np.ndarray,
                 pool_format: str = 'Reduced',
                 operators_list: list = []) -> None:
        

        self.nucleus: Nucleus = nucleus
        self.ref_state: np.ndarray = ref_state
        self.all_operators: list = self.nucleus.operators

        if pool_format == 'All':
            self.operator_pool = nucleus.operators
        elif pool_format == 'Reduced':
            self.operator_pool = self.reduce_operators()
        elif pool_format == 'ReducedII':
            self.operator_pool = self.reduce_operators_II()
        elif pool_format == 'ReducedIII':
            self.operator_pool = self.reduce_operators_III()
        elif pool_format == 'Only acting':
            self.operator_pool = self.only_acting_operators()
        elif pool_format == 'Custom':
            self.operator_pool = operators_list


        self.fcalls = 0
        self.count_fcalls: bool = False
        self.ansatz: np.ndarray = self.ref_state

    def reduce_operators(self) -> list:
        """Returns the list of non repeated operators used in the cluster"""

        operators = []
        all_ijkl = []
        for op in self.nucleus.operators:
            ijkl = op.ijkl
            klij = [ijkl[2], ijkl[3], ijkl[0], ijkl[1]]
            if klij not in all_ijkl:
                operators.append(op)
                all_ijkl.append(ijkl)
        return operators
    
    def reduce_operators_II(self) -> list:
        """Returns the list of non repeated operators used in the cluster"""

        operators = []
        all_matrix = []
        for op in self.nucleus.operators:
            matrix = op.matrix
            repeated = False
            for m in all_matrix:
                if np.allclose(matrix, -m) or np.allclose(matrix, m):
                    repeated = True
            if  repeated == False:
                operators.append(op)
                all_matrix.append(matrix)
        return operators
    
    def reduce_operators_III(self) -> list:
        """Returns the list of non repeated operators used in the cluster"""

        operators = []
        all_matrix = []
        for op in self.nucleus.operators:
            matrix = op.matrix
            repeated = False
            for m in all_matrix:
                if np.allclose(matrix, -m) or np.allclose(matrix, m):
                    repeated = True
            if  repeated == False and np.count_nonzero(matrix) <=2:
                operators.append(op)
                all_matrix.append(matrix)
        return operators
    

    def only_acting_operators(self) -> list:
        """Returns the list of operators that act on the reference state"""

        self.operator_pool = self.reduce_operators()
        operators = []
        for op in self.operator_pool:
            if np.allclose(op.matrix.dot(self.ref_state), np.zeros(len(self.ref_state))) == False:
                operators.append(op)
        return operators



class UCCAnsatz(Ansatz):
    """Class to define UCC ansätze"""

    def __init__(self, nucleus: Nucleus, 
                 ref_state: np.ndarray, 
                 T_n: int = 1,
                 pool_format: str = 'Only acting',
                 operators_list: list = []) -> None:
        
        super().__init__(nucleus=nucleus, ref_state=ref_state, pool_format=pool_format, operators_list=operators_list)
        
        self.T_n: int = T_n
        parameters: np.ndarray = np.zeros(len(self.operator_pool))
        self.build_ansatz(parameters)
        self.n_layers = len(self.operator_pool)


    def build_ansatz(self, parameters: list) -> np.ndarray:
        """Returns the ansatz"""

        ansatz = self.ref_state
        for t in range(self.T_n):
            for i, op in enumerate(self.operator_pool):
                ansatz = expm_multiply(parameters[i]/self.T_n * op.matrix, ansatz, traceA = 0.0)
        return ansatz
    
    def energy(self, parameters: list) -> float:
        """Returns the energy of the ansatz"""

        if len(parameters) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = self.build_ansatz(parameters)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            return E

    def sequential_energy(self, parameter: list) -> float:
        """Returns the energy of the ansatz"""

        if self.n_layers != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            self.ansatz = expm_multiply(self.operator_pool[self.n_layers-1].matrix*parameter[0], self.ansatz, traceA = 0.0)
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            return E

    def is_lie_algebra(self) -> bool:
        """Returns True if the operators form a Lie algebra"""
        operators = self.operator_pool
        for i in range(len(operators)):
            for j in range(i+1, len(operators)):
                commutator_ij = operators[i].matrix @ operators[j].matrix - operators[j].matrix @ operators[i].matrix
                if not any(np.allclose(commutator_ij, op.matrix) for op in operators) \
                and not any(np.allclose(commutator_ij, -op.matrix) for op in operators) \
                and not np.allclose(commutator_ij, np.zeros((self.nucleus.d_H, self.nucleus.d_H))):
                    print(commutator_ij)
                    return False
        return True
    
    def landscape(self, parameters: list, t: float, n: int) -> float:
        """Returns the energy of the ansatz with the parameter t in the n-th position"""

        parameters = np.array(parameters)
        parameters[n] = t
        return self.energy(parameters)


class ADAPTAnsatz(Ansatz):

    def __init__(self,
                 nucleus: Nucleus,
                 ref_state: np.ndarray,
                 pool_format: str = 'Reduced',
                 operators_list: list = []) -> None:
        
        super().__init__(nucleus, ref_state, pool_format, operators_list)

        self.added_operators: list = []
        self.minimum: bool = False
        self.E0: float = self.energy([])


    def build_ansatz(self, parameters: list) -> np.ndarray:
        """Returns the ansatz"""

        ansatz = self.ref_state
        for i,op in enumerate(self.added_operators):
            ansatz = expm_multiply(parameters[i]*op.matrix, ansatz, traceA = 0.0)
        return ansatz


    def energy(self, parameters: list) -> float:
        """Returns the energy of the ansatz"""

        if len(parameters) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = self.build_ansatz(parameters)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            return E


    def energy_one_step(self, parameter: list) -> float:
        
        if len(self.added_operators) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = expm_multiply(self.added_operators[-1].matrix*parameter[0], self.ansatz, traceA = 0.0)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            return E
    


    def choose_operator(self) -> tuple:
        """Selects the next operator based on its gradient and adds it to the list"""

        gradients = []
        sigma = self.nucleus.H.dot(self.ansatz)
        gradients = [abs(2*(sigma.conj().T.dot(op.matrix.dot(self.ansatz))).real) for op in self.operator_pool]
        max_gradient = max(gradients)
        max_operator = self.operator_pool[gradients.index(max_gradient)]
        return max_operator,max_gradient
