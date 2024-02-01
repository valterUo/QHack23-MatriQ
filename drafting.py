import numpy as np
from itertools import combinations
from utils import solve_bqm_in_leap
import concurrent.futures
import pickle

def flatten(data):
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield str(data)

def pair_to_index(x, dim):
    return dim*(x[0] - 1) + x[1] - 1

def get_standard_tensor(dim):
    initial_tensor = np.zeros((dim**2, dim**2, dim**2))
    for c1 in range(1, dim + 1):
        for c2 in range(1, dim + 1):
            c = (c1, c2)
            for a2 in range(1, dim + 1):
                a = (c1, a2)
                b = (a2, c2)
                initial_tensor[pair_to_index(a, dim)][pair_to_index(b, dim)][pair_to_index(c, dim)] = 1
    return initial_tensor

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
    print(sample)
    positive_linear_vars = []
    tensors = []
    for i in range(suggested_optimal):
        x, y, z = [], [], []
        for j in range(dim**2):
            x.append(sample[str(i) + "x" + str(j)])
            y.append(sample[str(i) + "y" + str(j)])
            z.append(sample[str(i) + "z" + str(j)])
        tensors.append([x,y,z])
    return tensors


# Generate three symbolic vectors of length 4
# and compute the dot product of the first two
# and the cross product of the last two.
# Print the results.

import dimod
import sympy as sym
from sympy import Array, tensorproduct, simplify


#for i, dim1 in enumerate(t):
#    for j, dim2 in enumerate(dim1):
#        for k, elem in enumerate(dim2):
#            expression = (initial_tensor[i][j][k] - elem)**2
#            expression = sym.expand(expression)
            
            # Replace every x**2 with x
#            symbols = expression.free_symbols
#            for symbol in symbols:
#                expression = expression.replace(symbol**2, symbol)
#            print(expression)
            
#            for term in expression.as_ordered_terms():
#                coeff, *vars = term.as_coeff_mul()
#                vars_tuple = tuple(sorted(str(v) for v in vars))
#                if vars_tuple in hubo:
#                    hubo[vars_tuple] += coeff
#                else:
#                    hubo[vars_tuple] = coeff

def process_element(i, j, k, elem, initial_tensor):
    expression = (int(initial_tensor[i][j][k]) - elem)**2
    expression = sym.expand(expression)
    
    symbols = expression.free_symbols
    for symbol in symbols:
        expression = expression.replace(symbol**2, symbol)

    local_hubo = {}
    offset = 0
    for term in expression.as_ordered_terms():
        coeff, vars = term.as_coeff_mul()
        if len(vars) == 0:
            offset += coeff
            continue
        vars = [str(v) for v in vars]
        vars_tuple = tuple(sorted(vars))
        if vars_tuple in local_hubo:
            local_hubo[vars_tuple] += coeff
        else:
            local_hubo[vars_tuple] = coeff

    return local_hubo, offset

if __name__ == '__main__':
    
    suggested_optimal = 7
    dim = 2
    xs = []
    ys = []
    zs = []
    for o in range(suggested_optimal):
        xs.append(Array(sym.symbols(str(o) + 'x0:4')))
        ys.append(Array(sym.symbols(str(o) + 'y0:4')))
        zs.append(Array(sym.symbols(str(o) + 'z0:4')))
        
        # Substitute every symbol to with an expression: x -> pos_x - neg_x
    xs_new = []
    for x_array in xs:
        xs_temp = []
        for x in x_array:
            new_x = x.subs(x, sym.Symbol('pos_' + str(x)) - sym.Symbol('neg_' + str(x)))
            xs_temp.append(new_x)
        xs_new.append(xs_temp)

    ys_new = []
    for y_array in ys:
        ys_temp = []
        for y in y_array:
            new_y = y.subs(y, sym.Symbol('pos_' + str(y)) - sym.Symbol('neg_' + str(y)))
            ys_temp.append(new_y)
        ys_new.append(ys_temp)
            
    zs_new = []
    for z_array in zs:
        zs_temp = []
        for z in z_array:
            new_z = z.subs(z, sym.Symbol('pos_' + str(z)) - sym.Symbol('neg_' + str(z)))
            zs_temp.append(new_z)
        zs_new.append(zs_temp)

    initial_tensor = get_standard_tensor(dim)

    t = tensorproduct(xs_new[0], tensorproduct(ys_new[0], zs_new[0]))
    for i in range(1, suggested_optimal):
        t -= tensorproduct(xs_new[i], tensorproduct(ys_new[i], zs_new[i]))

    hubo = {}
    total_offset = 0
    global_hubo = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        print("Starting the process pool")
        # Create a list of futures
        futures = [executor.submit(process_element, i, j, k, elem, initial_tensor)
                for i, dim1 in enumerate(t)
                for j, dim2 in enumerate(dim1)
                for k, elem in enumerate(dim2)]

        # Process the results as they complete
        for future in concurrent.futures.as_completed(futures):
            local_hubo, offset = future.result()
            total_offset += offset
            # Merge the local_hubo into global_hubo
            for key, value in local_hubo.items():
                if key in global_hubo:
                    global_hubo[key] += value
                else:
                    global_hubo[key] = value

    with open('hubo.pkl', 'wb') as f:
        result = {"hubo": global_hubo, "offset": total_offset}
        pickle.dump(result, f)