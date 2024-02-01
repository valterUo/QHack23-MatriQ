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

def square_negative_sum(hubo, variables, offset, aux_id):
    for v in variables:
        #v = tuple(sorted(list((flatten(var)))))
        if offset == 1:
            if v in hubo:
                hubo[v] -= 1
            else:
                hubo[v] = -1
        elif offset == 0:
            if v in hubo:
                hubo[v] += 1
            else:
                hubo[v] = 1
    combs = combinations(variables, 2)
    for pair in combs:
        v = tuple(sorted(list((flatten(pair)))))
        if v in hubo:
            hubo[v] += + 2
        else:
            hubo[v] = 2
    return hubo

# (1 - x - y - z)^2 = x^2 + y^2 + z^2 + 2xy + 2xz + 2yz - 2x - 2y - 2z + 1 = - x - y - z + 2xy + 2xz + 2yz + 1 = 1 - x - y -z # because 2 = 0 in F_2
# (0 - x - y - z)^2 = x^2 + y^2 + z^2 + 2xy + 2xz + 2yz = x + y + z + 2xy + 2xz + 2yz =  x + y + z # because 2 = 0 in F_2

# Options
# (2t - x - y - z)^2 = 4t^2 - 4tx - 4ty - 4tz + x^2 + 2xy + 2xz + y^2 + 2yz + z^2 = 4t - 4tx - 4ty - 4tz + x + 2xy + 2xz + y + 2yz + z
# (3t - x - y - z)^2 = 9t^2 - 9tx - 9ty - 9tz + x^2 + 2xy + 2xz + y^2 + 2yz + z^2 = 9t - 6tx - 6ty - 6tz + x + 2xy + 2xz + y + 2yz + z
# (4t - x - y - z)^2 = 16t^2 - 8tx - 8ty - 8tz + x^2 + 2xy + 2xz + y^2 + 2yz + z^2 = 16t - 8tx - 8ty - 8tz + x + 2xy + 2xz + y + 2yz + z
def square_negative_sum2(hubo, variables, offset, aux_id):
    #print(hubo)
    for var in variables:
        v = tuple(sorted(list((flatten(var)))))
        if offset == 1:
            if v in hubo:
                #print("1")
                hubo[v] = hubo[v] - 1
            else:
                #print("2")
                hubo[v] = -1
        elif offset == 0:
            if v in hubo:
                #print("3")
                hubo[v] = hubo[v] + 1
            else:
                #print("4")
                hubo[v] = 1
                
    if offset == 1:
        combs = combinations(variables, 2)
        for pair in combs:
            v = tuple(sorted(list(flatten(pair))))
            if v in hubo:
                #print("5")
                hubo[v] = hubo[v] + 2
            else:
                #print("6")
                hubo[v] = 2
    elif offset == 0:
        aux =  ("a" + str(aux_id),)
        if aux in hubo:
            #print("7")
            hubo[aux] = hubo[aux] + 4
        else:
            #print("8")
            hubo[aux] = 4
        variables.append(aux)
        #print(variables)
        aux_id = aux_id + 1
        combs = combinations(variables, 2)
        for pair in combs:
            v = tuple(sorted(list(flatten(pair))))
            if aux[0] in v:
                if v in hubo:
                    #print("9")
                    hubo[v] = hubo[v] - 4
                else:
                    #print("10")
                    hubo[v] = -4
            else:
                if v in hubo:
                    #print("11")
                    hubo[v] = hubo[v] + 2
                else:
                    #print("12")
                    hubo[v] = 2
    
    #print(hubo)   
    return hubo, aux_id

def variable_to_strassen(v, aux_ids = []):
    strassen_tensors = [[[0,0,0,1], [-1,0,1,0], [1,0,1,0]],
                        [[1,1,0,0], [0,0,0,1], [-1,1,0,0]],
                        [[-1,0,1,0], [1,1,0,0], [0,0,0,1]],
                        [[1,0,0,1], [1,0,0,1], [1,0,0,1]],
                        [[0,1,0,-1], [0,0,1,1], [1,0,0,0]],
                        [[1,0,0,0], [0,1,0,-1], [0,1,0,1]],
                        [[0,0,1,1], [1,0,0,0], [0,0,1,-1]]]
    
    mapping = {"x":0, "y":1, "z":2}
    print(v)
    if "a" in v:
            v = tuple(filter(lambda e: "a" not in e, v))
            
    if len(v) == 3:
        return np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])], 2)
    elif len(v) == 7:
        return np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])]*strassen_tensors[int(v[4])][mapping[v[5]]][int(v[6])], 2)
    elif len(v) == 11:
        return np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])]*strassen_tensors[int(v[4])][mapping[v[5]]][int(v[6])]*strassen_tensors[int(v[8])][mapping[v[9]]][int(v[10])], 2)
    elif len(v) == 15:
        return np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])]*strassen_tensors[int(v[4])][mapping[v[5]]][int(v[6])]*strassen_tensors[int(v[8])][mapping[v[9]]][int(v[10])]*strassen_tensors[int(v[12])][mapping[v[13]]][int(v[14])], 2)
    elif len(v) == 23:
        return np.mod(strassen_tensors[int(v[0])][mapping[v[1]]][int(v[2])]*strassen_tensors[int(v[4])][mapping[v[5]]][int(v[6])]*strassen_tensors[int(v[8])][mapping[v[9]]][int(v[10])]*strassen_tensors[int(v[12])][mapping[v[13]]][int(v[14])]*strassen_tensors[int(v[16])][mapping[v[17]]][int(v[18])]*strassen_tensors[int(v[20])][mapping[v[21]]][int(v[22])], 2)
    else:
        print("Error: variable_to_strassen")

    return 0
    
    

def get_test_strassen_test_tensor(bqm, aux_ids = []):
    variables = dict()
    for v in bqm.variables:
        variables[v] = variable_to_strassen(v)
    
    for v in bqm.variables:
        if "a" in v:
            #print(v)
            v2 = v.split("*")
            v2 = tuple(filter(lambda e: "a" in e, v2))
            v2 = v2[0]
            variables[v] = int(v2 in aux_ids)

    return variables

def get_full_higher_order_binary_optimization_problem(initial_tensor, dim, suggested_optimal, weight):
    hubo = dict()
    offset, aux_id = 0, 0
    aux_ids = []
    total_vars = set()
    for x in range(dim**2):
        for y in range(dim**2):
            for z in range(dim**2):
                # The tuple (x, y, z) represents the cube x*y*z which is the result of calculating tensor product between the vectors v_x, v_y and v_z
                variables = [(str(i) + "x" + str(x), str(i) + "y" + str(y), str(i) + "z" + str(z)) for i in range(suggested_optimal)]
                #total_vars = total_vars.union(set([str(i) + "x" + str(x) for i in range(suggested_optimal)]))
                #total_vars = total_vars.union(set([str(i) + "y" + str(y) for i in range(suggested_optimal)]))
                #total_vars = total_vars.union(set([str(i) + "z" + str(z) for i in range(suggested_optimal)]))
                #print(variables)
                offset += int(initial_tensor[x][y][z])
                hubo = square_negative_sum(hubo, variables, int(initial_tensor[x][y][z]), aux_id)
                
                if suggested_optimal == 8: # For bugging purposes
                    variables2 = [(str(i) + "x" + str(x), str(i) + "y" + str(y), str(i) + "z" + str(z)) for i in range(suggested_optimal)]
                    test_hubo, aux_id2 = square_negative_sum(dict(), variables2, int(initial_tensor[x][y][z]), 0)
                    test_bqm = dimod.make_quadratic(test_hubo, weight, dimod.BINARY)
                    test_tensor = get_test_strassen_test_tensor(test_bqm)
                    energy = test_bqm.energy(test_tensor) + int(initial_tensor[x][y][z])
                    if energy != 0:
                        aux_ids.append("a" + str(aux_id - 1))
                        test_tensor = get_test_strassen_test_tensor(test_bqm, ["a" + str(0)])
                        energy = test_bqm.energy(test_tensor) + int(initial_tensor[x][y][z])
                        if False: # For bugging purposes
                            print("Energy: ", energy)
                            for term in test_hubo:
                                if all([test_tensor[t] for t in term]):
                                    print(term, test_hubo[term])
    
    #print(hubo)
    #print("Total variables: ", len(total_vars))
    #print(len(hubo))
    bqm = dimod.make_quadratic(hubo, 10*weight, dimod.BINARY)
    bqm.offset = offset
    print("Offset: ", bqm.offset)
    #poly = dimod.BinaryPolynomial(hubo, dimod.BINARY)
    return bqm, hubo, aux_ids #, poly