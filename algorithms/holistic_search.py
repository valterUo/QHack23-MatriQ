import pickle

import dimod
from algorithm import Algorithm


class HolisticAlgorithm(Algorithm):
    
    def __init__(self, name, dim, suggested_optimal, hubo_file, normalize = True, bqm_transformation_strength = 2):
        self.name = name
        self.dim = dim
        self.suggested_optimal = suggested_optimal
        self.samplesets = {}
        
        with open(hubo_file, 'rb') as f:
            hubo_file_content = pickle.load(f)
            
        self.hubo, self.offset = hubo_file_content["hubo"], hubo_file_content["offset"]
        self.binary_polynomial = dimod.BinaryPolynomial.from_hubo(self.hubo, self.offset)
        
        if normalize:
            self.binary_polynomial.normalize()
            self.hubo, self.offset = self.binary_polynomial.to_hubo()
            
        self.bqm = dimod.make_quadratic(self.hubo, bqm_transformation_strength, dimod.BINARY)
        self.bqm.offset = self.offset
        self.bqm.normalize()
        print("Offset: ", self.bqm.offset)
        print("Linear: ", len(self.bqm.linear))
        print("Quadratic: ", len(self.bqm.quadratic))
        
        super().__init__(self.name, self.hubo, self.bqm)