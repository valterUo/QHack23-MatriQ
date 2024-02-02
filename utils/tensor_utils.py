import numpy as np


def pair_to_index(x, dim):
    return dim*(x[0] - 1) + x[1] - 1


def get_standard_tensor_rectangular(dim):
    result = np.full((dim**2, dim**2, dim**2), 0, dtype=np.int32)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                result[i * dim  + j][j * dim + k][k * dim + i] = 1
    return result


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