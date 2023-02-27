import dimod
from dimod.generators.constraints import combinations
from utils import *

# Constructs Non-Quadratic (Higher-Degree) Polynomial optimization problem and translates it to QUBO
# The output QUBO constains less variables than the first implementation in qubos.py 
# because the reduction HUBO -> QUBO is better


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


def towards_user_defined_full(initial_tensor, dim, suggested_optimal):
    cubic = dict()
    linear = dict()
    offset = 0.0
    
    for i in range(suggested_optimal):
        for x in range(dim**2):
            for y in range(dim**2):
                for z in range(dim**2):
                    coeff = 1
                    cube = (str(i) + "x" + str(x), str(i) + "y" + str(y), str(i) + "z" + str(z))
                    # Penalize cases when there is difference
                    if initial_tensor[x][y][z] != 0 and i == 0:
                        offset += 1
                        cubic[cube] = -1
                    else:
                        cubic[cube] = 1
    
    bqm = dimod.make_quadratic(cubic, offset, dimod.BINARY)
    return bqm