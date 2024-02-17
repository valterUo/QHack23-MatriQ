import pickle

import dimod
import numpy as np
from algorithms.algorithm import Algorithm


class HolisticAlgorithm(Algorithm):
    
    def __init__(self, name, dim, suggested_optimal, hubo_file, normalize = True, bqm_transformation_strength = 2):
        self.name = name
        self.dim = dim
        self.suggested_optimal = suggested_optimal
        self.samplesets = {}
        
        with open(hubo_file, 'rb') as f:
            hubo_file_content = pickle.load(f)
            
        self.hubo, self.offset = hubo_file_content["hubo"], float(hubo_file_content["offset"])
        print("Number of terms in hubo: ", len(self.hubo))
        self.binary_polynomial = dimod.BinaryPolynomial.from_hubo(self.hubo, self.offset)
        print("Number of variables in binary polynomial: ", len(self.binary_polynomial.variables))
        
        if normalize:
            self.binary_polynomial.normalize()
            self.hubo, self.offset = self.binary_polynomial.to_hubo()
            
        self.bqm = dimod.make_quadratic(self.hubo, bqm_transformation_strength, dimod.BINARY)
        self.bqm.offset = self.offset
        
        if normalize:
            self.bqm.normalize()
        
        print("Offset: ", self.bqm.offset)
        print("Linear: ", len(self.bqm.linear))
        print("Quadratic: ", len(self.bqm.quadratic))
        
        super().__init__(self.name, self.bqm, self.samplesets)
        
    def get_bqm(self):
        return self.bqm
    
    def get_ising(self):
        return self.ising
    
    def get_binary_polynomial(self):
        return self.binary_polynomial
        
    def get_samplesets(self):
        return self.samplesets
    
    def get_qubo_matrix(self):
        lin, (row, col, quad), offset, labels = self.bqm.to_numpy_vectors(sort_indices= True, sort_labels= True, return_labels=True)
        dim = len(lin)
        Q = np.zeros((dim, dim))
        np.fill_diagonal(Q, lin)
        Q[row, col] = quad
        return Q, labels, offset
    
    def get_ising_matrix(self):
        linear, quadratic, offset = self.bqm.to_ising()
        self.ising = dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.SPIN)
        self.ising.normalize()
        lin, (row, col, quad), offset, labels = self.ising.to_numpy_vectors(sort_indices= True, sort_labels= True, return_labels=True)
        dim = len(lin)
        J = np.zeros((dim, dim))
        np.fill_diagonal(J, lin)
        J[row, col] = quad
        return J, labels, self.ising.offset
    
    def get_optimal_energy(self):
        strassen_tensors = [[[0,0,0,1], [-1,0,1,0], [1,0,1,0]],
                        [[1,1,0,0], [0,0,0,1], [-1,1,0,0]],
                        [[-1,0,1,0], [1,1,0,0], [0,0,0,1]],
                        [[1,0,0,1], [1,0,0,1], [1,0,0,1]],
                        [[0,1,0,-1], [0,0,1,1], [1,0,0,0]],
                        [[1,0,0,0], [0,1,0,-1], [0,1,0,1]],
                        [[0,0,1,1], [1,0,0,0], [0,0,1,-1]]]

        variables_to_values = dict()
        for i in range(7):
            for j in range(2**2):
                if strassen_tensors[i][0][j] == 1:
                    variables_to_values["l_" + str(i) + "x" + str(j)] = 1
                    variables_to_values["r_" + str(i) + "x" + str(j)] = 0
                elif strassen_tensors[i][0][j] == -1:
                    variables_to_values["l_" + str(i) + "x" + str(j)] = 0
                    variables_to_values["r_" + str(i) + "x" + str(j)] = 1
                else:
                    variables_to_values["l_" + str(i) + "x" + str(j)] = 0
                    variables_to_values["r_" + str(i) + "x" + str(j)] = 0
                
                if strassen_tensors[i][1][j] == 1:
                    variables_to_values["l_" + str(i) + "y" + str(j)] = 1
                    variables_to_values["r_" + str(i) + "y" + str(j)] = 0
                elif strassen_tensors[i][1][j] == -1:
                    variables_to_values["l_" + str(i) + "y" + str(j)] = 0
                    variables_to_values["r_" + str(i) + "y" + str(j)] = 1
                else:
                    variables_to_values["l_" + str(i) + "y" + str(j)] = 0
                    variables_to_values["r_" + str(i) + "y" + str(j)] = 0
                        
                if strassen_tensors[i][2][j] == 1:
                    variables_to_values["l_" + str(i) + "z" + str(j)] = 1
                    variables_to_values["r_" + str(i) + "z" + str(j)] = 0
                elif strassen_tensors[i][2][j] == -1:
                    variables_to_values["l_" + str(i) + "z" + str(j)] = 0
                    variables_to_values["r_" + str(i) + "z" + str(j)] = 1
                else:
                    variables_to_values["l_" + str(i) + "z" + str(j)] = 0
                    variables_to_values["r_" + str(i) + "z" + str(j)] = 0
                                 
        for v in self.bqm.variables:
            if v not in variables_to_values:
                elems = v.split("*")
                multiplication_result = 1
                for e in elems:
                    multiplication_result = multiplication_result * variables_to_values[e]
                    variables_to_values[v] = multiplication_result
        energy_bqm = self.bqm.energy(variables_to_values)

        binary_polynomial = dimod.BinaryPolynomial.from_hubo(self.hubo, self.offset)
        energy = binary_polynomial.energy(variables_to_values)
        print("Because of Strasse's algorithm, we know the point which is the optimal solution:")
        print("BQM: ", energy_bqm)
        print("Binary Polynomial: ", energy)
        print("Offsets", self.bqm.offset, self.offset)
        
        
    def get_eigenvalues(self):
        from scipy import linalg as LA
        Q, labels, offset = self.get_qubo_matrix()
        eigenvalues, eigenvectors = LA.eigh(Q + offset * np.identity(Q.shape[0])) # offset * np.eye(Q.shape[0])
        return eigenvalues, eigenvectors, labels
    
    def get_singular_values(self):
        from scipy import linalg as LA
        Q, labels, offset = self.get_qubo_matrix()
        U, s, Vh = LA.svd(Q + offset * np.identity(Q.shape[0]))
        return U, s, Vh, labels
    
    
    def construct_all_tensors(self, result):
        tensors = []
        for i in range(self.suggested_optimal):
            x, y, z = [], [], []
            for j in range(self.dim**2):
                x.append(result["l_" + str(i) + "x" + str(j)] - result["r_" + str(i) + "x" + str(j)])
                y.append(result["l_" + str(i) + "y" + str(j)] - result["r_" + str(i) + "y" + str(j)])
                z.append(result["l_" + str(i) + "z" + str(j)] - result["r_" + str(i) + "z" + str(j)])
            tensors.append([x, y, z])
        return tensors
