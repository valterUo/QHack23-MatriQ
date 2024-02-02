import numpy as np
import dimod
from algorithm import Algorithm
from utils.qubo_utils import towards_user_defined_small
from utils.qubo_utils import process_result

from utils.tensor_utils import get_standard_tensor


class ModularAlgorithm(Algorithm):
    
    def __init__(self, dim = 2, suggested_optimal = 7):
        self.dim = dim
        self.suggested_optimal = suggested_optimal
        self.initial_tensor = get_standard_tensor(dim)
        self.origo = np.tensordot([0]*dim**2, np.tensordot([0]*dim**2, [0]*dim**2, axes=0), axes=0)
        
        self.strassen_tensors = [np.tensordot([0,0,0,1], np.tensordot([-1,0,1,0], [1,0,1,0], axes=0), axes=0),
          np.tensordot([1,1,0,0], np.tensordot([0,0,0,1], [-1,1,0,0], axes=0), axes=0),
           np.tensordot([-1,0,1,0], np.tensordot([1,1,0,0], [0,0,0,1], axes=0), axes=0),
           np.tensordot([1,0,0,1], np.tensordot([1,0,0,1], [1,0,0,1], axes=0), axes=0),
          np.tensordot([0,1,0,-1], np.tensordot([0,0,1,1], [1,0,0,0], axes=0), axes=0),
           np.tensordot([1,0,0,0], np.tensordot([0,1,0,-1], [0,1,0,1], axes=0), axes=0),
           np.tensordot([0,0,1,1], np.tensordot([1,0,0,0], [0,0,1,-1], axes=0), axes=0)]
        
        self.strassen_vectors = [[[0,0,0,1], [-1,0,1,0], [1,0,1,0]],
                        [[1,1,0,0], [0,0,0,1], [-1,1,0,0]],
                        [[-1,0,1,0], [1,1,0,0], [0,0,0,1]],
                        [[1,0,0,1], [1,0,0,1], [1,0,0,1]],
                        [[0,1,0,-1], [0,0,1,1], [1,0,0,0]],
                        [[1,0,0,0], [0,1,0,-1], [0,1,0,1]],
                        [[0,0,1,1], [1,0,0,0], [0,0,1,-1]]]
        
        self.bqm = dimod.BinaryQuadraticModel({}, {}, 0, dimod.BINARY)
        self.samplesets = {}
        super().__init__("strasse", self.bqm, self.samplesets)
    
    
    def run_modular_algorithm(self):
        tensor = (self.initial_tensor - self.strassen_tensors[0] - self.strassen_tensors[1] - self.strassen_tensors[2] - self.strassen_tensors[3]) % 2

        # Moving towards origo
        while(True):
            self.bqm = towards_user_defined_small(tensor, self.origo, self.dim)
            sampleset = self.solve_with_Greedy()
            sample = sampleset.first.sample
            x1, y1, z1 = process_result(sample, 2)
            print(x1, y1, z1)
            tensor = (tensor - np.tensordot(x1, np.tensordot(y1, z1, axes=0), axes=0)) % 2
            if np.count_nonzero(tensor.flatten()) == 0:
                print("End")
                break
                
        tensor = (self.initial_tensor + self.strassen_tensors[0] + self.strassen_tensors[1] + self.strassen_tensors[2]) % 2
        # Moving towards standard matrix multiplication i.e. the naive method
        while(True):
            self.bqm = towards_user_defined_small(tensor, self.initial_tensor, self.dim)
            sampleset = self.solve_with_Greedy()
            sample = sampleset.first.sample
            x1, y1, z1 = process_result(sample, 2)
            print(x1, y1, z1)
            tensor = (tensor - np.tensordot(x1, np.tensordot(y1, z1, axes=0), axes=0)) % 2
            if np.array_equal(tensor, self.initial_tensor):
                print("End")
                break
        