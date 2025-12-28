import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp

# Function to compute the final eigenstate of the qubits
def find_eigenstate(qubo, ansatz):
    # Convert QUBO to Hamiltonian and define device
    op, offset = qubo.to_ising()
    num_wires = op.num_qubits
    wires = list(range(num_wires))

    hamiltonian = op
    H_matrix = hamiltonian.to_matrix()
    H = qml.Hermitian(H_matrix, wires=wires)

    dev = qml.device("default.qubit", wires=num_wires, shots=1000)

    # Define the QNode with the ansatz
    @qml.qnode(dev, interface="jax")
    def qnode(params):
        ansatz(params, wires=wires)
        return qml.expval(H)

    # Set up optimization settings
    learning_rate = 5e-4
    num_steps = 100
    init_params = np.random.uniform(low=0, high=2 * np.pi, size=(2 * num_wires,))
    
    # JIT compile gradient and QNode
    grad_fn = jax.jit(jax.grad(qnode))
    qnode_jit = jax.jit(qnode)

    # Optimization loop
    params = init_params
    for step in range(num_steps):
        cost = qnode_jit(params)
        grads = grad_fn(params)
        params = params - learning_rate * grads
    
    # Define function to obtain the final eigenstate
    def final_state(params):
        ansatz(params, wires=wires)
        return [qml.sample(qml.PauliZ(i)) for i in range(num_wires)]

    # Measure qubit states using optimal parameters
    qubit_states = final_state(params)
    qubit_states_np = np.array(qubit_states)
    qubit_states_np[qubit_states_np == -1] = 0  # Map -1 to 0

    # Take the most common measurement outcome for each qubit
    final_qubit_states = [np.bincount(qubit_states_np[i]).argmax() for i in range(len(qubit_states_np))]
    
    return final_qubit_states


def hardware_efficient_ansatz(params, wires):
    num_wires = len(wires)
    for i in range(num_wires):
        qml.RY(params[i], wires=wires[i])
        qml.RZ(params[num_wires + i], wires=wires[i])
    for i in range(num_wires - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

def single_layer_ansatz(params, wires):
    num_wires = len(wires)
    for i in range(num_wires):
        qml.RX(params[i], wires=wires[i])
        qml.RZ(params[num_wires + i], wires=wires[i])
    for i in range(num_wires - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def qaoa_ansatz(params, wires):
    num_wires = len(wires)
    for i in range(num_wires):
        qml.RY(params[i], wires=wires[i])
    for i in range(num_wires - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    for i in range(num_wires):
        qml.RY(params[num_wires + i], wires=wires[i])


def alternating_layer_ansatz(params, wires):
    num_wires = len(wires)
    for i in range(num_wires):
        qml.RX(params[i], wires=wires[i])
        qml.RY(params[num_wires + i], wires=wires[i])
    for i in range(num_wires - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

