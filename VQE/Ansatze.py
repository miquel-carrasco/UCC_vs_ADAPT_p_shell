import numpy as np
from .Nucleus import Nucleus, TwoBodyExcitationOperator
from scipy.linalg import expm
from scipy.optimize import minimize
import random


class Ansatz():
    """Class to define ansätze"""

    def __init__(self, nucleus: Nucleus, ref_state: np.ndarray) -> None:
        self.nucleus: Nucleus = nucleus
        self.ref_state: np.ndarray = ref_state
        self.operators: list[TwoBodyExcitationOperator] = self.nucleus.operators

        self.fcalls = 0
        self.op_applied=0
        self.count_fcalls: bool = False
        self.ansatz: np.ndarray = self.ref_state

class UCCAnsatz(Ansatz):
    """Class to define UCC ansätze"""

    def __init__(self, nucleus: Nucleus, 
                 ref_state: np.ndarray, 
                 T_n: int = 1, 
                 cluster_format: str = 'Only acting', 
                 operators_list: list[TwoBodyExcitationOperator] = []) -> None:
        
        super().__init__(nucleus, ref_state)
        
        if cluster_format == 'All':
            self.operators = nucleus.operators
        elif cluster_format == 'Reduced':
            self.operators = self.reduce_operators()
        elif cluster_format == 'Only acting':
            self.operators = self.only_acting_operators()
        elif cluster_format == 'Custom':
            self.operators = operators_list
        self.T_n: int = T_n

        parameters: np.ndarray = np.zeros(len(self.operators))
        self.build_ansatz(parameters)


    def reduce_operators(self) -> list[TwoBodyExcitationOperator]:
        """Returns the list of non repeated operators used in the cluster"""

        operators = []
        all_ijkl = []
        for op in self.operators:
            ijkl = op.ijkl
            klij = [ijkl[2], ijkl[3], ijkl[0], ijkl[1]]
            if klij not in all_ijkl:
                operators.append(op)
                all_ijkl.append(ijkl)
        return operators
    

    def only_acting_operators(self) -> list[TwoBodyExcitationOperator]:
        """Returns the list of operators that act on the reference state"""

        self.operators = self.reduce_operators()
        operators = []
        for op in self.operators:
            if np.allclose(op.matrix @ self.ref_state, np.zeros(len(self.ref_state))) == False:
                operators.append(op)
        return operators


    def build_ansatz(self, parameters: list[float]) -> np.ndarray:
        """Returns the ansatz"""

        ansatz = self.ref_state
        for t in range(self.T_n):
            for i, op in enumerate(self.operators):
                ansatz = expm(parameters[i]/self.T_n * op.matrix) @ ansatz
        return ansatz
    
    def energy(self, parameters: np.ndarray) -> float:
        """Returns the energy of the ansatz"""

        ansatz = self.build_ansatz(parameters)
        if self.count_fcalls == True:
            self.fcalls += 1
        return ansatz.conj().T @ self.nucleus.H @ ansatz

    def is_lie_algebra(self) -> bool:
        """Returns True if the operators form a Lie algebra"""
        operators = self.operators
        for i in range(len(operators)):
            for j in range(i+1, len(operators)):
                commutator_ij = operators[i].matrix @ operators[j].matrix - operators[j].matrix @ operators[i].matrix
                if not any(np.allclose(commutator_ij, op.matrix) for op in operators) \
                and not any(np.allclose(commutator_ij, -op.matrix) for op in operators) \
                and not np.allclose(commutator_ij, np.zeros((self.nucleus.d_H, self.nucleus.d_H))):
                    print(commutator_ij)
                    return False
        return True
    
    def lanscape(self, parameters: list[float], t: float, n: int) -> float:
        """Returns the energy of the ansatz with the parameter t in the n-th position"""

        parameters = np.array(parameters)
        parameters[n] = t
        return self.energy(parameters)


class ADAPTAnsatz(Ansatz):

    def __init__(self, nucleus: Nucleus, ref_state: np.ndarray) -> None:
        super().__init__(nucleus, ref_state)

        self.added_operators:list[TwoBodyExcitationOperator] = []
        self.minimum: bool = False


    def build_ansatz(self, parameters: list[float]) -> np.ndarray:
        """Returns the ansatz"""

        ansatz = self.ref_state
        for i,op in enumerate(self.added_operators):
            ansatz = expm(parameters[i]*op.matrix) @ ansatz
        return ansatz
    
    def energy(self, parameters: list[float]) -> float:
        """Returns the energy of the ansatz"""

        if self.count_fcalls == True:
            self.fcalls += 1
        new_ansatz = self.build_ansatz(parameters)
        return 1000*new_ansatz.conj().T @ self.nucleus.H @ new_ansatz

    def choose_operator(self) -> tuple[TwoBodyExcitationOperator, float]:
        """Selects the next operator based on its gradient and adds it to the list"""

        gradients = []
        gradients = [abs(self.ansatz.conj().T @ op.commutator @ self.ansatz) for op in self.operators]
        max_gradient = max(gradients)
        max_operator = self.operators[gradients.index(max_gradient)]
        return max_operator,max_gradient

