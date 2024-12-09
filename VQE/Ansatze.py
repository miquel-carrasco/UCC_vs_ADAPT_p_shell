import numpy as np
from VQE.Nucleus import Nucleus, TwoBodyExcitationOperator
from scipy.linalg import expm
from  scipy.sparse.linalg import expm_multiply



class Ansatz():
    """
    Parent class to define ansätze for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        all_operators (list): List of all the avaliable two-body excitation operators for a given nucleus.
        operator_pool (list): List of operators used in the ansatz.
        fcalls (int): Number of function calls of a VQE procedure.
        count_fcalls (bool): If True, the number of function calls during a VQE procedure is counted.
        ansatz (np.ndarray): Ansatz state.
    
    Methods:
        reduce_operators: Returns the list of operators excluding the repeated excitations.
        only_acting_operators: Returns the list of operators, only including the ones that have a non-zero action on the ref. state.
    """
    def __init__(self,
                 nucleus: Nucleus,
                 ref_state: np.ndarray,
                 pool_format: str = 'Reduced',
                 operators_list: list = []) -> None:
        """
        Initialization of the Ansatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool. Available formats ['All', 'Reduced', 'Only acting', 'Custom'].
            operators_list (list): List of operators to be used in the ansatz, in case the pool format is 'Custom'.
        """
        self.nucleus = nucleus
        self.ref_state = ref_state
        self.all_operators = self.nucleus.operators
        if pool_format == 'All':
            self.operator_pool = nucleus.operators
        elif pool_format == 'Reduced':
            self.operator_pool = self.reduce_operators()
        elif pool_format == 'Only acting':
            self.operator_pool = self.only_acting_operators()
        elif pool_format == 'Custom':
            self.operator_pool = operators_list
        self.fcalls = 0
        self.count_fcalls = False
        self.ansatz = self.ref_state

    
    def reduce_operators(self) -> list:
        """
        Returns the list of operators excluding the repeated excitations.

        Returns:
            list: List of operators.
        """
        operators = []
        all_matrix = []
        for op in self.nucleus.operators:
            if np.allclose(op.matrix, np.zeros((self.nucleus.d_H, self.nucleus.d_H))) == False:
                matrix = op.matrix
                repeated = False
                for m in all_matrix:
                    if np.allclose(matrix, -m) or np.allclose(matrix, m):
                        repeated = True
                if  repeated == False:
                    operators.append(op)
                    all_matrix.append(matrix)
        return operators
   

    def only_acting_operators(self) -> list:
        """
        Returns the list of operators, only including the ones that have a non-zero action on the ref. state.

        Returns:
            list: List of operators.
        """
        self.operator_pool = self.reduce_operators()
        operators = []
        for op in self.operator_pool:
            if np.allclose(op.matrix.dot(self.ref_state), np.zeros(len(self.ref_state))) == False:
                operators.append(op)
        return operators


class UCCAnsatz(Ansatz):
    """
    Child Ansatz class to define the Unitary Coupled Cluster ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        T_n (int): Number of Trotter steps.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        ansatz (np.ndarray): Ansatz state.
        n_layers (int): Number of layers of the ansatz.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, 
                      after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
    """
    def __init__(self, nucleus: Nucleus, 
                 ref_state: np.ndarray, 
                 T_n: int = 1,
                 pool_format: str = 'Reduced',
                 operators_list: list = []) -> None:
        
        super().__init__(nucleus=nucleus, ref_state=ref_state, pool_format=pool_format, operators_list=operators_list)
        
        self.T_n: int = T_n
        parameters: np.ndarray = np.zeros(len(self.operator_pool))
        self.build_ansatz(parameters)
        self.n_layers = len(self.operator_pool)


    def build_ansatz(self, parameters: list) -> np.ndarray:
        """
        Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            np.ndarray: Ansatz state.
        """
        ansatz = self.ref_state
        for t in range(self.T_n):
            for i, op in enumerate(self.operator_pool):
                ansatz = expm_multiply(parameters[i]/self.T_n * op.matrix, ansatz, traceA = 0.0)
        return ansatz
    

    def energy(self, parameters: list) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.        
        """
        if len(parameters) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = self.build_ansatz(parameters)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            return E
        

class ADAPTAnsatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus: Nucleus,
                 ref_state: np.ndarray,
                 pool_format: str = 'Reduced',
                 operators_list: list = []) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state, pool_format, operators_list)
        self.added_operators = []
        self.minimum = False
        self.E0 = self.energy([])


    def build_ansatz(self, parameters: list) -> np.ndarray:
        """
        Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.

        Returns:
            np.ndarray: Ansatz state.        
        """
        ansatz = self.ref_state
        for i,op in enumerate(self.added_operators):
            ansatz = expm_multiply(parameters[i]*op.matrix, ansatz, traceA = 0.0)
        return ansatz


    def energy(self, parameters: list) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
        if len(parameters) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = self.build_ansatz(parameters)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            return E


    def choose_operator(self) -> tuple:
        """
        Returns the next operator and its gradient, after an ADAPT iteration.

        Returns:
            TwoBodyExcitationOperator: Next operator.
            float: Gradient of the next operator.
        """

        gradients = []
        sigma = self.nucleus.H.dot(self.ansatz)
        gradients = [abs(2*(sigma.conj().T.dot(op.matrix.dot(self.ansatz))).real) for op in self.operator_pool]
        max_gradient = max(gradients)
        max_operator = self.operator_pool[gradients.index(max_gradient)]
        if len(self.added_operators) == 1:
            op = [o for o in self.operator_pool if o.ijkl == [3, 6, 4, 8]][0]
            gradient = abs(2*(sigma.conj().T.dot(op.matrix.dot(self.ansatz))).real)
            print(gradient)
            print(max_gradient)
        return max_operator,max_gradient