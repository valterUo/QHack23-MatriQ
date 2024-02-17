import copy
import pickle
import numpy as np
import dimod
from algorithms.algorithm import Algorithm
from utils.qubo_utils import move_tensors_closer
from utils.qubo_utils import process_result
from utils.tensor_utils import get_standard_tensor

def pair_to_index(x, dim):
    return dim*(x[0] - 1) + x[1] - 1

# The standard multiplication tensor for 2 x 2 matrices
# AlphaTensor paper uses the symmetric version of this tensor
# which is not compatible with the current implementation 
def get_standard_tensor_for_strassen(dim_n, dim_m, dim_p):
    initial_tensor = np.zeros((dim_n*dim_m, dim_m*dim_p, dim_p*dim_n))
    for c1 in range(1, dim_n + 1):
        for c2 in range(1, dim_m + 1):
            c = (c1, c2)
            for a2 in range(1, dim_p + 1):
                a = (c1, a2)
                b = (a2, c2)
                initial_tensor[pair_to_index(a, dim_n)][pair_to_index(b, dim_m)][pair_to_index(c, dim_p)] = 1
    return initial_tensor


class ModularAlgorithm(Algorithm):
    
    def __init__(self, name, dim_n = 2, dim_m = None, dim_p = None, field = "r"):
        self.name = name
        self.dim_n = dim_n
        self.dim_m = dim_m
        self.dim_p = dim_p
        self.field = field
        self.total_rank = 0
        
        if dim_m is None:
            self.dim_m = dim_n
        if dim_p is None:
            self.dim_p = dim_n
            
        self.initial_tensor = get_standard_tensor(self.dim_n, self.dim_m, self.dim_p)
        
        if self.name == "2,2,2": #or self.name == "3,3,3":
            self.initial_tensor = get_standard_tensor_for_strassen(self.dim_n, self.dim_m, self.dim_p)
            
        self.origo = np.zeros((self.dim_n*self.dim_m, self.dim_m*self.dim_p, self.dim_p*self.dim_n))
        
        if self.field == "r":
            with open("files//ordered_tensor_factorizations_r.pkl", "rb") as f:
                self.factorizations = pickle.load(f)     
        elif self.field == "f2":
            with open("files//ordered_tensor_factorizations_f2.pkl", "rb") as f:
                self.factorizations = pickle.load(f)
                
        if self.name in self.factorizations:
            
            self.tensors = self.factorizations[self.name]["tensors"]
            self.increasing_order = self.factorizations[self.name]["increasing_order"]
            self.decreasing_order = self.factorizations[self.name]["decreasing_order"]
            self.suggested_optimal = self.factorizations[self.name]["rank"]
            
        elif self.name == "2,2,2":
            
            self.strassen_tensors = [np.tensordot([0,0,0,1], np.tensordot([-1,0,1,0], [1,0,1,0], axes=0), axes=0),
                                    np.tensordot([1,1,0,0],  np.tensordot([0,0,0,1], [-1,1,0,0], axes=0), axes=0),
                                    np.tensordot([-1,0,1,0], np.tensordot([1,1,0,0], [0,0,0,1], axes=0), axes=0),
                                    np.tensordot([1,0,0,1],  np.tensordot([1,0,0,1], [1,0,0,1], axes=0), axes=0),
                                    np.tensordot([0,1,0,-1], np.tensordot([0,0,1,1], [1,0,0,0], axes=0), axes=0),
                                    np.tensordot([1,0,0,0],  np.tensordot([0,1,0,-1], [0,1,0,1], axes=0), axes=0),
                                    np.tensordot([0,0,1,1],  np.tensordot([1,0,0,0], [0,0,1,-1], axes=0), axes=0)]
            
            self.strassen_vectors = [[[0,0,0,1], [-1,0,1,0], [1,0,1,0]],
                                    [[1,1,0,0], [0,0,0,1], [-1,1,0,0]],
                                    [[-1,0,1,0], [1,1,0,0], [0,0,0,1]],
                                    [[1,0,0,1], [1,0,0,1], [1,0,0,1]],
                                    [[0,1,0,-1], [0,0,1,1], [1,0,0,0]],
                                    [[1,0,0,0], [0,1,0,-1], [0,1,0,1]],
                                    [[0,0,1,1], [1,0,0,0], [0,0,1,-1]]]
            
            self.suggested_optimal = 7
        
        self.bqm = dimod.BinaryQuadraticModel({}, {}, 0, dimod.BINARY)
        self.samplesets = {}
        super().__init__(self.name, self.bqm, self.samplesets)
    
    
    def run_modular_algorithm(self):
        high_energy_tensor = self.initial_tensor
        
        if self.name == "2,2,2":
            if self.field == "f2":
                high_energy_tensor = np.mod(self.initial_tensor - self.strassen_tensors[0] - self.strassen_tensors[1] - self.strassen_tensors[2], 2)
                flip_tensor = self.strassen_tensors[4]
            elif self.field == "r":
                high_energy_tensor = self.initial_tensor - self.strassen_tensors[0] - self.strassen_tensors[1] - self.strassen_tensors[2] - self.strassen_tensors[3]
                flip_tensor = self.strassen_tensors[4]
        else:
            #print(self.increasing_order)
            #print(self.decreasing_order)
            
            for i in range(len(self.increasing_order) - 1):
                
                if self.field == "r":
                    high_energy_tensor -= self.tensors[i]
                elif self.field == "f2":
                    high_energy_tensor = np.mod(high_energy_tensor - self.tensors[i], 2)
                    
            flip_tensor = self.tensors[len(self.increasing_order) - 1]
        
        tensor = copy.deepcopy(high_energy_tensor)
        
        print("Number of non-zeros initially: ", np.count_nonzero(tensor))
        # Moving towards origo from the high energy point
        while(True):
            
            self.bqm = move_tensors_closer(tensor,
                                            self.initial_tensor,
                                            self.dim_n, 
                                            self.dim_m, 
                                            self.dim_p)
            sampleset = self.solve_with_Gurobi()
            sample = sampleset["result"]
            print("Energy: ", sampleset["energy"])
            #sample = sampleset.first.sample
            x1, y1, z1 = self.construct_vectors_from_sample(sample, 
                                                            self.dim_n, 
                                                            self.dim_m, 
                                                            self.dim_p)
            print(x1, y1, z1)
            
            if self.field == "r":
                tensor = tensor - np.tensordot(x1, np.tensordot(y1, z1, axes=0), axes=0)
            elif self.field == "f2":
                tensor = np.mod(tensor - np.tensordot(x1, np.tensordot(y1, z1, axes=0), axes=0), 2)
            
            print("Number of nonzeros: ", np.count_nonzero(tensor))
            
            self.total_rank += 1
            
            with open("files//tensor" + self.name +".txt", "w") as f:
                f.write(str(x1) + "\n" + str(y1) + "\n" + str(z1) + "\n" + str(tensor))
            
            if np.array_equal(tensor, self.initial_tensor):
                print("END")
                break
            
                
        #tensor = (self.initial_tensor + self.strassen_tensors[0] + self.strassen_tensors[1] + self.strassen_tensors[2]) % 2
        if self.name == "2,2,2":
            if self.field == "r":
                tensor = high_energy_tensor - flip_tensor
            elif self.field == "f2":
                tensor = np.mod(high_energy_tensor - flip_tensor, 2)
        else:
            if self.field == "r":
                tensor = high_energy_tensor - flip_tensor
            elif self.field == "f2":
                tensor = np.mod(high_energy_tensor - flip_tensor, 2)
        
        print("Number of non-zeros initially: ", np.count_nonzero(tensor))
        
        # Moving towards standard matrix multiplication from the high energy point
        while(True):
            
            self.bqm = move_tensors_closer(tensor, 
                                            self.origo,
                                            self.dim_n, 
                                            self.dim_m, 
                                            self.dim_p)
            sampleset = self.solve_with_Gurobi()
            sample = sampleset["result"]
            print("Energy: ", sampleset["energy"])
            #sample = sampleset.first.sample
            x1, y1, z1 = self.construct_vectors_from_sample(sample, 
                                                            self.dim_n, 
                                                            self.dim_m, 
                                                            self.dim_p)
            
            print(x1, y1, z1)
            
            if self.field == "r":
                tensor = tensor - np.tensordot(x1, np.tensordot(y1, z1, axes=0), axes=0)
            elif self.field == "f2":
                tensor = np.mod(tensor - np.tensordot(x1, np.tensordot(y1, z1, axes=0), axes=0), 2)
            
            print("Number of nonzeros: ", np.count_nonzero(tensor))
            self.total_rank += 1
            with open("files//tensor" + self.name +".txt", "w") as f:
                f.write(str(x1) + "\n" + str(y1) + "\n" + str(z1) + "\n" + str(tensor))        
            
            if np.array_equal(tensor, self.origo):
                print("END")
                break
            
        print("Total rank: ", self.total_rank)
            
            
    def construct_vectors_from_sample(self, sample, dim_n, dim_m, dim_p):
        x = np.zeros(dim_n*dim_m)
        y = np.zeros(dim_m*dim_p)
        z = np.zeros(dim_p*dim_n)
        for i in range(dim_n*dim_m):
            x[i] = sample["xl" + str(i)] #- sample["xr" + str(i)]
        for i in range(dim_m*dim_p):
            y[i] = sample["yl" + str(i)] #- sample["yr" + str(i)]
        for i in range(dim_p*dim_n):
            z[i] = sample["zl" + str(i)] #- sample["zr" + str(i)]
        return x, y, z
        