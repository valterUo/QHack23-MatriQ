from itertools import combinations

import dimod
import numpy as np
from general_utils import flatten

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

def construct_all_tensors(sample, dim, suggested_optimal):
    tensors = []
    for i in range(suggested_optimal):
        x, y, z = [], [], []
        for j in range(dim**2):
            x.append(sample["l_" + str(i) + "x" + str(j)] - sample["r_" + str(i) + "x" + str(j)])
            y.append(sample["l_" + str(i) + "y" + str(j)] - sample["r_" + str(i) + "y" + str(j)])
            z.append(sample["l_" + str(i) + "z" + str(j)] - sample["r_" + str(i) + "z" + str(j)])
        tensors.append([x,y,z])
    return tensors

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


def process_result(sample, dim):
    x, y, z = [], [], []
    error = validate_sample(sample, dim)
    print("Number of errors:", error)
    if error > 1000:
        return [], [], []
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