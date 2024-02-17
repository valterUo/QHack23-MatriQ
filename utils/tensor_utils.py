import numpy as np

# This standard matrix multiplication tensor generator should match with the 
# standard matrix multiplication tensors used in AlphaTensor paper
def get_standard_tensor(dim_n, dim_m = None, dim_p = None):
    if dim_m is None:
        dim_m = dim_n
    if dim_p is None:
        dim_p = dim_n
    initial_tensor = np.zeros((dim_n*dim_m, dim_m*dim_p, dim_p*dim_n))
    for i in range(dim_n):
        for j in range(dim_m):
            for k in range(dim_p):
                initial_tensor[i*dim_m + j][j*dim_p + k][k*dim_n + i] = 1
    return initial_tensor