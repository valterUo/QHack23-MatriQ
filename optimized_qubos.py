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

# (x - y - z)^2 = x^2 + y^2 + z^2 - 2xy - 2xz + 2yz
def square_negative_sum(variables):
    hubo = dict()
    offset = 0
    pos_var = variables[0]
    
    if type(pos_var) == int or type(pos_var) == float:
        offset = pos_var
    
    for var in variables:
        if type(var) != int and type(var) != float:
            hubo[var] = 1
        
    combs = combinations(variables, 2)
    for pair in combs:
        if pair[0] == pos_var:
            hubo[tuple(flatten(pair[1]))] = -2
        elif pair[1] == pos_var:
            hubo[tuple(flatten(pair[0]))] = -2
        else:   
            hubo[tuple(flatten(pair))] = 2
            
    return hubo, offset
    

def towards_user_defined_full(initial_tensor, dim, suggested_optimal):
    offset = 0.0
    full_bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    
    for x in range(dim**2):
        for y in range(dim**2):
            for z in range(dim**2):
                variables = []
                if initial_tensor[x][y][z] != 0:
                        variables.append(1)
                for i in range(suggested_optimal):
                    cube = (str(i) + "x" + str(x), str(i) + "y" + str(y), str(i) + "z" + str(z))
                    variables.append(cube)
                    
                hubo, offset = square_negative_sum(variables)
                bqm = dimod.make_quadratic(hubo, offset, dimod.BINARY)
                full_bqm.update(bqm)
                
    return full_bqm