import dimod
import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np


def print_probs(qaoa_circuit, wires, params):
    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        qaoa_circuit([gamma, alpha])
        return qml.probs(wires=wires)


    probs = probability_circuit(params[0], params[1])

    plt.bar(range(2 ** len(wires)), probs)
    plt.show()


def get_qml_hamiltionian(bqm):
    var_to_z_number = {}
    for i, var in enumerate(bqm.linear):
        var_to_z_number[var] = i
        
    coeffs = []
    obs = []
    
    for var in bqm.iter_linear():
        z_number = var_to_z_number[var[0]]
        obs.append(qml.PauliZ(z_number))
        coeffs.append(float(var[1]))
    for var in bqm.iter_quadratic():
        z_number1 = var_to_z_number[var[0]]
        z_number2 = var_to_z_number[var[1]]
        obs.append(qml.PauliZ(z_number1) @ qml.PauliZ(z_number2))
        coeffs.append(float(var[2]))
    
    obs.append(qml.Identity(0))
    coeffs.append(bqm.offset)
    H = qml.Hamiltonian(coeffs, obs)
    return (H, qaoa.x_mixer(var_to_z_number.values()))


# See https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html
def construct_qaoa_and_optimize(ocean_bqm, device = "default.qubit", optimizer = "GradientDescentOptimizer", steps = 100):
    cost, mix = get_qml_hamiltionian(ocean_bqm)
    
    #print("Cost Hamiltonian:", cost)
    #print("Mixer Hamiltonian:", mix)
    
    wires = range(len(ocean_bqm.linear))
    dev = None
    if device == "Braket":
        dev = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", wires=wires)
    else:
        dev = qml.device(device, wires=wires)
        
    depth = len(ocean_bqm.linear)
    
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost)
        qaoa.mixer_layer(alpha, mix)

    def qaoa_circuit(params):
        qml.broadcast(qml.Hadamard, wires, 'single')
        qml.layer(qaoa_layer, depth, params[0], params[1])

    @qml.qnode(dev)
    def cost_function(params):
        qaoa_circuit(params)
        return qml.expval(cost)
    
    if optimizer == "GradientDescentOptimizer":
        optimizer = qml.GradientDescentOptimizer()
    elif optimizer == "SPSA":
        optimizer = qml.SPSAOptimizer(maxiter=steps)
    elif optimizer == "QNSPSA":
        optimizer = qml.QNSPSAOptimizer()
    elif optimizer == "Adagrad":
        optimizer = qml.AdagradOptimizer()
        
    params = np.array([[0.5]*len(wires), [0.5]*len(wires)], requires_grad=True)
    
    if optimizer == "QNSPSA":
        for i in range(steps):
            params, energy = optimizer.step_and_cost(cost_function, params)
            print("Step:", i, "Params: ", params, "Energy: ", energy)
    else:
        for i in range(steps):
            print("Step:", i)
            params = optimizer.step(cost_function, params)
            print("Params: ", params)

    print("Optimal parameters:", params)
    
    return params, qaoa_circuit, wires