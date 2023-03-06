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

# (1 - x - y)^2 = x^2 + y^2 + 2xy - 2x - 2y + 1 = - x - y + 2xy + 1 = 1 - x - y # because 2 = 0 in F_2
# (0 - x - y - z)^2 = x^2 + y^2 + z^2 + 2xy + 2xz + 2yz = x + y + z + 2xy + 2xz + 2yz =  x + y + z # because 2 = 0 in F_2
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
        
    #combs = combinations(variables, 2)
    #if offset == 1:
    #    for pair in combs:
   #         v = tuple(sorted(list(flatten(pair))))
  #          if v in hubo:
  #              hubo[v] = hubo[v] + 2
  #          else:
  #              hubo[v] = 2
  #  else:
   #     for pair in combs:
    #        v = tuple(sorted(list(flatten(pair))))
     #       if v in hubo:
      #          hubo[v] = hubo[v] - 2
       #     else:
        #        hubo[v] = -2
        
            
    return hubo

def get_test_strassen_test_tensor(bqm):

    strassen_tensors = [[[0,0,0,1], [-1,0,1,0], [1,0,1,0]],
                        [[1,1,0,0], [0,0,0,1], [-1,1,0,0]],
                        [[-1,0,1,0], [1,1,0,0], [0,0,0,1]],
                        [[1,0,0,1], [1,0,0,1], [1,0,0,1]],
                        [[0,1,0,-1], [0,0,1,1], [1,0,0,0]],
                        [[1,0,0,0], [0,1,0,-1], [0,1,0,1]],
                        [[0,0,1,1], [1,0,0,0], [0,0,1,-1]]]

    mapping = {"x":0, "y":1, "z":2}
    variables = dict()
    for v in bqm.variables:
        if len(v) == 3:
            variables[v] = np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])], 2)
        elif len(v) == 7:
            variables[v] = np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])]*strassen_tensors[int(v[4])][mapping[v[5]]][int(v[6])], 2)
        elif len(v) == 11:
            variables[v] = np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])]*strassen_tensors[int(v[4])][mapping[v[5]]][int(v[6])]*strassen_tensors[int(v[8])][mapping[v[9]]][int(v[10])], 2)
        elif len(v) == 15:
            variables[v] = np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])]*strassen_tensors[int(v[4])][mapping[v[5]]][int(v[6])]*strassen_tensors[int(v[8])][mapping[v[9]]][int(v[10])]*strassen_tensors[int(v[12])][mapping[v[13]]][int(v[14])], 2)

    return variables

def towards_user_defined_full(initial_tensor, dim, suggested_optimal):
    hubo = dict()
    offset = 0
    weight = 1
    hubos = [[[dict()]*(dim**2)]*(dim**2)]*(dim**2)
    for x in range(dim**2):
        for y in range(dim**2):
            for z in range(dim**2):
                variables = [(str(i) + "x" + str(x), str(i) + "y" + str(y), str(i) + "z" + str(z)) for i in range(suggested_optimal)]
                offset += int(initial_tensor[x][y][z])
                hubo = square_negative_sum(hubo, variables, int(initial_tensor[x][y][z]))
                
                test_hubo = square_negative_sum(dict(), variables, int(initial_tensor[x][y][z]))
                test_bqm = dimod.make_quadratic(test_hubo, weight, dimod.BINARY)
                test_tensor = get_test_strassen_test_tensor(test_bqm)
                energy = test_bqm.energy(test_tensor) + int(initial_tensor[x][y][z])
                if energy != 0:
                    print("Energy: ", energy)
                #    for e in hubo:
                #        if len(e) == 3:
                #            print(e)
                    
                #    print(initial_tensor[x][y][z]) 
                #    print(hubo)
                    
                    for term in test_hubo:
                        if all([test_tensor[t] for t in term]):
                            print(term, test_hubo[term])

                #    for key in test_tensor:
                #        if len(key) == 3:
                #            print(key, test_tensor[key])
    
    bqm = dimod.make_quadratic(hubo, weight, dimod.BINARY)
    bqm.offset = offset
    poly = dimod.BinaryPolynomial(hubo, dimod.BINARY)
    return bqm, poly