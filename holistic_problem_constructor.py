import concurrent.futures
import pickle
import sympy as sym
from sympy import Array, tensorproduct
from utils.tensor_utils import get_standard_tensor

"""
This script is used to construct the HUBO (Higher Order Unconstrained Binary Optimization) problem for the given tensor decomposition optimization problem.

The algorithm uses parallel processing to speed up sympy calculations and thus the functionality is separated from the main algorithms.

It uses the holistic approach: it construct a HUBO that encodes the minimization problem of the full decomposition instead of a single step (compared to the modular approach).

Later this HUBO problem can be converted into QUBO problem and solved using D-Wave quantum annealer or other classical solvers.

We need to find the exact minimum (0) or otherwise the solution is not a valid tensor decomposition. On the other hand, there are multiple different parameter configurations that 
produce the optimal decomposition.

Run this file to produce the hubo.pkl file which can be imported to the solvers.
"""

def calculate_local_hubo(tensor_elem, constant_elem):
    expression = sym.Pow(int(constant_elem) - tensor_elem, 2)
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
            print("Offset: ", offset)
            continue
        vars = [str(v) for v in vars]
        vars_tuple = tuple(sorted(vars))
        if vars_tuple in local_hubo:
            local_hubo[vars_tuple] += int(coeff)
        else:
            local_hubo[vars_tuple] = int(coeff)

    return local_hubo, offset

# Substitute every symbol to with an expression: x -> l_x - r_x (i.e. left - right)
def substitute_variables(vars):
    xs_new = []
    for x_array in vars:
        xs_temp = []
        for x in x_array:
            new_x = x.subs(x, sym.Symbol('l_' + str(x)) - sym.Symbol('r_' + str(x)))
            xs_temp.append(new_x)
        xs_new.append(xs_temp)
    return xs_new


if __name__ == '__main__':
    
    hubo_file_name = "files//hubo_333.pkl"
    suggested_optimal = 23
    
    # Multiply matrices dim_n x dim_m and dim_m x dim_p
    dim_n = 3
    dim_m = 3
    dim_p = 3
    xs, ys, zs = [], [], []
    total_offset = 0
    global_hubo = {}
    
    for i in range(suggested_optimal):
        xs.append(Array(sym.symbols(str(i) + 'x0:' + str(dim_n*dim_m))))
        ys.append(Array(sym.symbols(str(i) + 'y0:' + str(dim_m*dim_p))))
        zs.append(Array(sym.symbols(str(i) + 'z0:' + str(dim_p*dim_n))))
        
    xs_new = substitute_variables(xs)
    ys_new = substitute_variables(ys)
    zs_new = substitute_variables(zs)

    initial_tensor = get_standard_tensor(dim_n, dim_m, dim_p)

    t = tensorproduct(xs_new[0], tensorproduct(ys_new[0], zs_new[0]))
    for i in range(1, suggested_optimal):
        t = t + tensorproduct(xs_new[i], tensorproduct(ys_new[i], zs_new[i]))

    with concurrent.futures.ProcessPoolExecutor(max_workers=7) as executor:
        print("Starting the process pool")
        # Create a list of futures
        futures = []
        for i in range(dim_n*dim_m):
            for j in range(dim_m*dim_p):
                for k in range(dim_p*dim_n):
                    tensor_elem = t[i][j][k]
                    #print("Tensor element: ", tensor_elem)
                    constant_elem = initial_tensor[i][j][k]
                    futures.append(executor.submit(calculate_local_hubo, tensor_elem, constant_elem))

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

    with open(hubo_file_name, 'wb') as f:
        result = {"hubo": global_hubo, "offset": total_offset}
        pickle.dump(result, f)