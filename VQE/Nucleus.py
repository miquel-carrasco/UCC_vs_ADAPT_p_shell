import numpy as np
from numpy import linalg as la
import os
import warnings
import scipy
from scipy.sparse import lil_matrix, csc_matrix
import scipy.sparse
from numba import jit, cuda

class TwoBodyExcitationOperator():
    "Class to define an antihermitian operator corresponding to a two-body excitation."

    def __init__(self, label: int, H2b: float, ijkl: list[int], matrix: lil_matrix, commutator: np.ndarray) -> None:
        self.label = label
        self.H2b = H2b
        self.ijkl = ijkl
        self.matrix = matrix
        self.commutator = commutator
        

class Nucleus():
    """Class to define a nucleus with its Hamiltonian, eigenvalues and eigenvectors, 
        angular momentum and other properties."""

    def __init__(self, nuc_name: str, J: int, M: int=0) -> None:
        "Initializes the nucleus with its name, angular momentum and magnetic quantum number."
        self.name = nuc_name
        self.J = J
        self.M = M
        self.data_folder = os.path.join(f'nuclei/{self.name}_data')
        self.states = self.states()
        self.H = self.hamiltonian_matrix()
        self.eig_val, self.eig_vec = la.eigh(self.H)
        self.operators = self.sparse_operators()
    

    def hamiltonian_matrix(self) -> np.array:
        "Returns the hamiltonian matrix of the nucleus."
        file_path = os.path.join(self.data_folder, f'{self.name}.dat')
        H = np.zeros((self.d_H, self.d_H))
        H_data = np.loadtxt(file_path,delimiter=' ', dtype=float)
        for line in H_data:
            H[int(line[0]), int(line[1])] = line[2]
        return H
    
    def states(self) -> list[tuple]:

        states = []
        mb_path = os.path.join(self.data_folder, f'mb_basis_2.dat')
        file = open(mb_path, 'r')
        self.d_H = int(file.readline().strip())
        mb_data = np.loadtxt(mb_path, dtype=str, delimiter=' ',skiprows=1)
        for m in mb_data:
            sp_labels = []
            for i in range(1, len(m)):
                label = int(m[i].replace(',','').replace('(','').replace(')',''))
                sp_labels.append(int(label))
            states.append(tuple(sp_labels))
        return states
    
    def operators_list(self) -> list[TwoBodyExcitationOperator]:
        """Returns the list of ALL antihermitian operators corresponding
            to two-body excitations. Each operator is represented by a TwoBodyExcitationOperator object."""
        
        matrices_folder = os.path.join(self.data_folder, 'mats')

        H2b_path = os.path.join(self.data_folder, f'H2b.dat')
        H2b_data = np.loadtxt(H2b_path, dtype=str)
        values = []
        indices = []
        for h in H2b_data:
            values.append(float(h[0]))
            indices.append(h[1] + ' ' + h[2] + ' ' + h[3] + ' ' + h[4])
        H2b_dictionary = dict(zip(indices, values))

        labels_path = os.path.join(self.data_folder, 'op_labels.dat')
        labels_data = np.loadtxt(labels_path, dtype=str)
        operators = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for labels in labels_data:
                label = int(labels[0])
                i = int(labels[1].replace('^','').replace('[','').replace(']',''))
                j = int(labels[2].replace('^','').replace('[','').replace(']',''))
                k = int(labels[3].replace('^','').replace('[','').replace(']',''))
                l = int(labels[4].replace('^','').replace('[','').replace(']',''))
                ijkl=[i,j,k,l]
                
                matrix_path = os.path.join(matrices_folder, f'mat{label}.dat')
                matrix_data = np.loadtxt(matrix_path)
                if len(matrix_data) != 0:
                    operator_matrix = np.zeros((self.d_H, self.d_H))
                    for a in matrix_data:
                        operator_matrix[int(a[0]), int(a[1])] = a[2]
                    H2b = H2b_dictionary[f'{ijkl[0]} {ijkl[1]} {ijkl[2]} {ijkl[3]}']
                    commutator = self.H @ operator_matrix - operator_matrix @ self.H
                    operators.append(TwoBodyExcitationOperator(label, H2b, ijkl, operator_matrix, commutator))

        return operators
    
    def sparse_operators(self) -> list[TwoBodyExcitationOperator]:
        "Returns the list of operators with sparse matrices."

        operators = []

        H2b_path = os.path.join(self.data_folder, f'H2b.dat')
        H2b_data = np.loadtxt(H2b_path, dtype=str)
        label = 1
        for h in H2b_data:
            indices = [int(h[1]), int(h[2]), int(h[3]), int(h[4])]
            if indices[0] < indices[1] and indices[2] < indices[3]:
                operator_matrix = lil_matrix((self.d_H, self.d_H))
                for state in self.states:
                    new_state = self.excitation_numbers(state, indices)
                    if new_state in self.states:
                        H2b = float(h[0])
                        column = self.states.index(state)
                        row = self.states.index(new_state)
                        this_excitation = lil_matrix((self.d_H, self.d_H))
                        this_excitation[row, column] = 1
                        operator_matrix += this_excitation
                        operator_matrix += -this_excitation.T
                        commutator = self.H @ operator_matrix - operator_matrix @ self.H
                if operator_matrix.count_nonzero() != 0:
                    operators.append(TwoBodyExcitationOperator(label, H2b, indices, operator_matrix.toarray(), commutator))
                    label += 1
        
        return operators
    

    def excitation_numbers(self, state: tuple, indices: list[int]) -> tuple:

        if indices[2] in state and indices[3] in state:
            new_state = list(state)
            for i in range(len(new_state)):
                if new_state[i]==indices[2]:
                    new_state[i] = indices[0]
                    break
            for j in range(len(new_state)):
                if new_state[j]==indices[3]:
                    new_state[j] = indices[1]
                    break
            new_state.sort()
            return tuple(new_state)
        else:
            return tuple()
            
            
if __name__=='__main__':
    He8 = Nucleus('He8', 0)
    list = [op.matrix for op in He8.operators if op.ijkl==[7,8,6,9]]
    mat = list[0]
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i,j] != 0:
                print(i,j,mat[i,j])