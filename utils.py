import numpy as np
import dimod
from dwave.system import LeapHybridSampler
from hybrid.reference import KerberosSampler
import neal
import greedy


def append_linear_safe(variable, value, linear_dict):
    if variable in linear_dict.keys():
        linear_dict[variable] = linear_dict[variable] + value
    else:
        linear_dict[variable] = value

        
def append_quadratic_safe(variable, value, quadratic_dict):
    if variable in quadratic_dict.keys():
        quadratic_dict[variable] = quadratic_dict[variable] + value
    else:
        quadratic_dict[variable] = value

        
def print_solution(sample):
    positive_solution = []
    for varname, value in sample.items():
        if value == 1 and type(varname) != tuple:
            positive_solution.append(varname)
            print(varname, value)
            
            
def get_initial_tensor_old(dim, original_multiplication):
    initial_tensor = np.zeros((dim**2, dim**2, dim**2))
    for t in original_multiplication:
        initial_tensor[t[0]][t[1]][t[2]] = 1
    return initial_tensor


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

def post_process(sample):
    new_sample = {}
    for k in sample:
        if "," in k:
            new_sample[eval(k)] = sample[k]
        else:
            new_sample[k] = sample[k]
    return new_sample

def construct_uvw_tensor(dim):
    indices_1 = []
    indices_2 = []
    indices_3 = []
    for x in range(dim**2):
        for y in range(dim**2):
            for z in range(dim**2):
                indices_1.append(((x, y), z))
                indices_2.append(((x, z), y))
                indices_3.append(((y, z), x))
    return indices_1, indices_2, indices_3

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


def solve_bqm_in_leap(bqm, sampler = "Kerberos"):
    #bqm.normalize()
    if sampler == "Kerberos":
        sampler = KerberosSampler()
        sampleset = sampler.sample(bqm, max_iter=7000, convergence=9, qpu_params={'label': 'Matrix multiplication'}, sa_reads=100000, sa_sweeps=1000000, qpu_reads=1)
    elif sampler == "LeapHybrid":
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm)
    elif sampler == "Simulated":
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=10000)
    elif sampler == "Greedy":
        sampler = greedy.SteepestDescentSolver()
        sampleset = sampler.sample(bqm, num_reads = 1000000)
    elif sampler == "Exact":
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)
    sample = sampleset.first.sample
    energy = sampleset.first.energy
    print("Energy: ", energy)
    return sample, energy, sampleset


def vectorize(u):
    vectors = []
    for i in range(len(u[0])):
        vector = []
        for us in u:
            vector.append(us[i])
        vectors.append(vector)
    return vectors
                