import numpy as np
from numpy import linalg as la
import os
import warnings

class TwoBodyExcitationOperator():
    "Class to define an antihermitian operator corresponding to a two-body excitation."

    def __init__(self, label: int, H2b: float, ijkl: list[int], matrix: np.ndarray, commutator: np.ndarray) -> None:
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
        self.H = self.hamiltonian_matrix()
        self.d_H = len(self.H)
        self.eig_val, self.eig_vec = la.eigh(self.H)
        self.operators= self.operators_list()
    

    def hamiltonian_matrix(self) -> np.array:
        "Returns the hamiltonian matrix of the nucleus."
        file_path = os.path.join(self.data_folder, f'{self.name}.dat')
        H_data = np.loadtxt(file_path)
        d_H = int(H_data[-1,0])+1
        H = H_data[:,2].reshape(d_H,d_H)
        return H
    
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







