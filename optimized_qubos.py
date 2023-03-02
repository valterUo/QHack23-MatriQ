import dimod
from dimod.generators.constraints import combinations
from utils import *
from itertools import combinations

# Constructs Non-Quadratic (Higher-Degree) Polynomial optimization problem and translates it to QUBO
# The output QUBO constains less variables than the first implementation in qubos.py 
# because the reduction HUBO -> QUBO is better


def flatten(data):
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data
        

def towards_user_defined_small(initial_tensor, target_tensor, dim):
    cubic = dict()
    offset = 0.0
    indices = []
    for x in range(dim**2):
        for y in range(dim**2):
            for z in range(dim**2):
                coeff = 1
                cube = ("x" + str(x), "y" + str(y), "z" + str(z))
                # Penalize cases when there is difference
                if initial_tensor[x][y][z] != target_tensor[x][y][z]:
                    offset += 1
                    cubic[cube] = -1
                else:
                    cubic[cube] = 1
                    
    bqm = dimod.make_quadratic(cubic, offset, dimod.BINARY)
    return bqm

# (1 - x - y)^2 = x^2 + y^2 + 2xy - 2x - 2y + 1 = - x - y + 2xy + 1
# (0 - x - y)^2 = x^2 + y^2 + 2xy = x + y + 2xy
def square_negative_sum(hubo, variables, offset):
    for var in variables:
        v = tuple(sorted(list((flatten(var)))))
        if offset == 1:
            if v in hubo:
                hubo[v] = hubo[v] - 1
            else:
                hubo[v] = -1
        else:
            if v in hubo:
                hubo[v] = hubo[v] + 1
            else:
                hubo[v] = 1
        
    combs = combinations(variables, 2)
    for pair in combs:
        v = tuple(sorted(list(flatten(pair))))
        if v in hubo:
            hubo[v] = hubo[v] + 2
        else:
            hubo[v] = 2
            
    return hubo
    

def towards_user_defined_full(initial_tensor, dim, suggested_optimal):
    hubo = dict()
    offset = 0
    for x in range(dim**2):
        for y in range(dim**2):
            for z in range(dim**2):
                variables = [(str(i) + "x" + str(x), str(i) + "y" + str(y), str(i) + "z" + str(z)) for i in range(suggested_optimal)]
                offset += int(initial_tensor[x][y][z])
                hubo = square_negative_sum(hubo, variables, int(initial_tensor[x][y][z]))
    #bqm = dimod.make_quadratic(hubo, 1000, dimod.BINARY)
    hqm = dimod.BinaryPolynomial(hubo, dimod.BINARY)
    reduced, mapping = dimod.reduce_binary_polynomial(hqm)
    bqm = dimod.BinaryQuadraticModel(dict(), dict(), offset, dimod.BINARY)
    for quad in reduced:
        u, v = quad[0]
        bqm.add_quadratic(u, v, quad[1])
    #reduced, quadratic = dimod.reduce_binary_polynomial(dimod.BinaryPolynomial(hubo, dimod.BINARY))
    #print(linear)
    #for t in reduced:
    #    print(t)
        #if len(t[0]) == 1:
        #    print(t)
    #print(len(quadratic))
    #bqm = dimod.BinaryQuadraticModel(reduced, offset, dimod.BINARY)
    return bqm