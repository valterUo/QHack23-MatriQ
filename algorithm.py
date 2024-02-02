import time
import cplex
import numpy as np
import dimod
from dwave.system import LeapHybridSampler
from hybrid.reference import KerberosSampler
from dwave.samplers import TabuSampler, SteepestDescentSolver, SimulatedAnnealingSampler, TreeDecompositionSolver

class Algorithm:
    
    def __init__(self, name, bqm, samplesets = {}):
        self.name = name
        self.bqm = bqm
        self.samplesets = samplesets
        
    def solve_with_Kerberos(self):
        sampler = KerberosSampler()
        sampleset = sampler.sample(self.bqm, max_iter=10000, convergence=10, 
                                   qpu_params={'label': 'Matrix multiplication'}, 
                                   sa_reads=100000, sa_sweeps=1000000, qpu_reads=1000)
        self.samplesets["Kerberos"] = sampleset
        energy = sampleset.first.energy
        print("Energy: ", energy)
        return sampleset
    
    def solve_with_LeapHybrid(self):
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(self.bqm)
        self.samplesets["LeapHybrid"] = sampleset
        energy = sampleset.first.energy
        print("Energy: ", energy)
        return sampleset
    
    def solve_with_Simulated(self):
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(self.bqm, beta_range=[.1, 4.2], beta_schedule_type='linear')
        self.samplesets["Simulated"] = sampleset
        energy = sampleset.first.energy
        print("Energy: ", energy)
        return sampleset
    
    def solve_with_Greedy(self):
        sampler = SteepestDescentSolver()
        sampleset = sampler.sample(self.bqm, num_reads = 100000)
        self.samplesets["Greedy"] = sampleset
        energy = sampleset.first.energy
        print("Energy: ", energy)
        return sampleset
    
    def solve_with_Exact(self):
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(self.bqm)
        self.samplesets["Exact"] = sampleset
        energy = sampleset.first.energy
        print("Energy: ", energy)
        return sampleset
    
    def solve_with_Tabu(self):
        sampler = TabuSampler()
        sampleset = sampler.sample(self.bqm, num_reads = 10000)
        self.samplesets["Tabu"] = sampleset
        energy = sampleset.first.energy
        print("Energy: ", energy)
        return sampleset
    
    def solve_with_Tree(self):
        sampler = TreeDecompositionSolver()
        sampleset = sampler.sample(self.bqm, num_reads = 10000)
        self.samplesets["Tree"] = sampleset
        energy = sampleset.first.energy
        print("Energy: ", energy)
        return sampleset
    
    def qubo_to_lp(self, identifier):
        cplex_problem = cplex.Cplex()
        cplex_problem.objective.set_sense(cplex_problem.objective.sense.minimize)
        variable_symbols = [str(var) for var in self.bqm.variables]
        cplex_problem.variables.add(names=variable_symbols, 
                                            types=[cplex_problem.variables.type.binary]*len(variable_symbols))

        linear_coeffs = self.bqm.linear
        obj_list = [(str(name), coeff) for name, coeff in linear_coeffs.items()]
        cplex_problem.objective.set_linear(obj_list)

        quadratic_coeffs = self.bqm.quadratic
        obj_list = [(str(name[0]), str(name[1]), coeff) for name, coeff in quadratic_coeffs.items()]
        cplex_problem.objective.set_quadratic_coefficients(obj_list)
        lp_file = "linear_program_files//" + str(identifier) + ".lp"
        cplex_problem.write(lp_file)
        return cplex_problem
    
    def solve_with_CPLEX(self, print_log = False):
        cplex_problem = self.qubo_to_lp(self.name)
        
        if not print_log:
            cplex_problem.set_log_stream(None)
            cplex_problem.set_error_stream(None)
            cplex_problem.set_warning_stream(None)
            cplex_problem.set_results_stream(None)
        
        # Allow multiple threads
        cplex_problem.parameters.threads.set(4)
        # Print progress
        cplex_problem.parameters.mip.display.set(2)
        time_start = time.time()
        cplex_problem.solve()
        time_end = time.time()
        elapsed_time = time_end - time_start
        status = cplex_problem.solution.get_status_string()
        result = cplex_problem.solution.get_values()
        variables = cplex_problem.variables.get_names()
        value = cplex_problem.solution.get_objective_value() + self.bqm.offset
        self.samplesets["cplex"] = { "status": status, "energy": value, 
                                    "time": elapsed_time, "result": dict(zip(variables, result)) }
        return self.samplesets["cplex"]