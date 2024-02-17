from itertools import combinations

import dimod
import numpy as np
import sympy as sym
from utils.general_utils import flatten

def square_negative_sum(hubo, variables, offset):
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
            hubo[v] += 2
        else:
            hubo[v] = 2
    return hubo

def construct_all_tensors(sample, dim_n, dim_m, dim_p, suggested_optimal):
    tensors = []
    for i in range(suggested_optimal):
        x, y, z = [], [], []
        for j in range(dim_n*dim_m):
            x.append(sample["l_" + str(i) + "x" + str(j)] - sample["r_" + str(i) + "x" + str(j)])
        for j in range(dim_m*dim_p):
            y.append(sample["l_" + str(i) + "y" + str(j)] - sample["r_" + str(i) + "y" + str(j)])
        for j in range(dim_p*dim_n):
            z.append(sample["l_" + str(i) + "z" + str(j)] - sample["r_" + str(i) + "z" + str(j)])
        tensors.append([x,y,z])
    return tensors

def move_tensors_closer(initial_tensor, target_tensor, dim_n, dim_m, dim_p):
    local_hubo = {}
    offset = 0
    
    for x in range(dim_n*dim_m):
        for y in range(dim_m*dim_p):
            for z in range(dim_p*dim_n):
                left_cube = ["xl" + str(x), "yl" + str(y), "zl" + str(z)]
                right_cube = ["xr" + str(x), "yr" + str(y), "zr" + str(z)]
                left_cube_symbols = [sym.symbols(l) for l in left_cube]
                right_cube_symbols = [sym.symbols(r) for r in right_cube]
                difference = [left - right for left, right in zip(left_cube_symbols, right_cube_symbols)]
                #product = sym.Mul(*difference)
                product = sym.Mul(*left_cube_symbols)
                #print(product)
                product = sym.expand(product)
                # Penalize cases when there is difference
                init = initial_tensor[x][y][z]
                target = target_tensor[x][y][z]
                if init != target:
                    product = 1 - product
                    if False:
                        if init == 0 and target == 1:
                            product = 1 - product
                        elif init == 1 and target == 0:
                            product = product
                        elif init == -1 and target == 0:
                            product = 1 - product
                        elif init == 0 and target == -1:
                            product = product
                        elif init == -2 and target == 0:
                            continue
                        else:
                            print("Target tensor has a value ", target , ". Init has a value ", init)
                else:
                    product = product
                    # If the values are the same, we should not change the tensor
                    # Thus we add triple xl*yl*zl
                    #product = sym.expand(sym.Pow(product, 2))
                    #symbols = product.free_symbols
                    #for symbol in symbols:
                    #    product = product.replace(symbol**2, symbol)
                
                for term in product.as_ordered_terms():
                    coeff, vars = term.as_coeff_mul()
                    if len(vars) == 0:
                        offset += coeff
                        continue
                    vars = [str(v) for v in vars]
                    vars_tuple = tuple(sorted(vars))
                    if vars_tuple in local_hubo:
                        local_hubo[vars_tuple] += int(coeff)
                    else:
                        local_hubo[vars_tuple] = int(coeff)
                   
    bqm = dimod.make_quadratic(local_hubo, 8, dimod.BINARY)
    bqm.offset = offset
    #print("Number of variables: ", len(bqm.variables))
    return bqm

def validate_sample(sample, dim):
    error_rate = 0
    for i in range(dim**2):
        if "x" + str(i) in sample.keys():
            if sample["x" + str(i)] == 1:
                for j in range(dim**2):
                    if "y" + str(j) in sample.keys():
                        if sample["y" + str(j)] == 1:
                            if ("x" + str(i), "y" + str(j)) in sample.keys():
                                if sample[("x" + str(i), "y" + str(j))] != 1:
                                    error_rate += 1
                        if "z" + str(j) in sample.keys():
                            if sample["z" + str(j)] == 1:
                                if ("x" + str(i), "z" + str(j)) in sample.keys():
                                    if sample[("x" + str(i), "z" + str(j))] != 1:
                                        error_rate += 1
    for i in range(dim**2):
        if "y" + str(i) in sample.keys():
            if sample["y" + str(i)] == 1:
                for j in range(dim**2):
                    if "z" + str(j) in sample.keys():
                        if sample["z" + str(j)] == 1:
                            if ("y" + str(i), "z" + str(j)) in sample.keys():
                                if sample[("y" + str(i), "z" + str(j))] != 1:
                                    error_rate += 1
    for k in sample:
        if "," in k:
            t = eval(k)
            if sample[k] == 1:
                if sample[t[0]] != 1:
                    error_rate += 1
                elif sample[t[1]] != 1:
                    error_rate += 1
    return error_rate


def process_result(sample, dim_n, dim_m, dim_p):
    x, y, z = [], [], []
    #error = validate_sample(sample, dim)
    #print("Number of errors:", error)
    #if error > 1000:
    #    return [], [], []
    for i in range(dim**2):
        if "x" + str(i) in sample.keys():
            x.append(sample["x" + str(i)])
        else:
            x.append(1)
        if "y" + str(i) in sample.keys():
            y.append(sample["y" + str(i)])
        else:
            y.append(1)
        if "z" + str(i) in sample.keys():
            z.append(sample["z" + str(i)])
        else:
            z.append(1)
    if all([i == 0 for i in x]) and all([i == 0 for i in y]) and all([i == 0 for i in z]):
        return [], [], []
    if all([i == 1 for i in x]) and all([i == 1 for i in y]) and all([i == 1 for i in z]):
        return [], [], []
    return np.array(x), np.array(y), np.array(z)