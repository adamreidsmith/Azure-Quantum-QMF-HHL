import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation, QFT, CUGate
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider


A = np.array([[1, -1 / 3], [-1 / 3, 1]])
b = np.array([[0], [1]])

eigs = np.linalg.eig(A)
for i, val in enumerate(eigs.eigenvalues):
    print(f'Eigenvector {eigs.eigenvectors[:, i]} with eignevalue {val}')

n = 2
n_b = 1

# Set up the registers
ancilla = QuantumRegister(1)
clock = QuantumRegister(n)
input = QuantumRegister(n_b)
measurement = ClassicalRegister(2)
hhl = QuantumCircuit(ancilla, clock, input, measurement)

# Prepare the input register
sp = StatePreparation(b.flatten())
hhl.append(sp, input)
hhl.barrier()

# Perform phase estimation
U = CUGate(theta=np.pi / 2, phi=-np.pi / 2, lam=np.pi / 2, gamma=3 * np.pi / 4)
U2 = CUGate(theta=np.pi, phi=np.pi, lam=0, gamma=0)
hhl.h(clock)
hhl.barrier()
hhl.append(U, [clock[0], *input])
hhl.append(U2, [clock[1], *input])
hhl.barrier()
hhl.compose(QFT(2), clock, inplace=True)
hhl.barrier()

# Apply ancilla rotation
hhl.cry(np.pi, clock[0], ancilla)
hhl.cry(np.pi / 3, clock[1], ancilla)
hhl.barrier()

# Uncompute the phase estimation
U2_inv = U2
U_inv = CUGate(theta=np.pi / 2, phi=np.pi / 2, lam=-np.pi / 2, gamma=-3 * np.pi / 4)
hhl.compose(QFT(2, inverse=True), clock, inplace=True)
hhl.barrier()
hhl.append(U2_inv, [clock[1], *input])
hhl.append(U_inv, [clock[0], *input])
hhl.barrier()
hhl.h(clock)
hhl.barrier()

# Measure the ancilla
hhl.measure(ancilla, 0)

# Measure x
hhl.measure(input, 1)

# Connect to Azure Quantum
workspace = Workspace(
    resource_id="/subscriptions/d570a97a-c638-4dd4-a437-4916bfc1bbe8/resourceGroups/GHGQuantum-rg/providers/Microsoft.Quantum/Workspaces/GHG-Quantum-Workspace",
    location="westus",
)
provider = AzureQuantumProvider(workspace)
# backend = provider.get_backend('ionq.qpu')
# backend = provider.get_backend('quantinuum.qpu.h1-1')
# backend = provider.get_backend('rigetti.qpu.ankaa-2')
backend = provider.get_backend('ionq.simulator')
for back in provider.backends():
    print(back.name())

# from qiskit_ibm_runtime import QiskitRuntimeService  # fmt: skip
# runtime = QiskitRuntimeService()
# backend = runtime.get_backend('ibm_sherbrooke')
# # backend = AerSimulator()

# Transpile the HHL circuit
hhl_transpiled = transpile(hhl, backend, optimization_level=3, approximation_degree=0.9)
print(hhl_transpiled.depth(), hhl_transpiled.num_qubits)

# Estimate costs. Available for IonQ and Quantinuum backends
# print(backend.estimate_cost(hhl_transpiled, shots=1000))

# Run the circuit and plot the results
job = backend.run(hhl_transpiled, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)
plot_histogram(counts)

# Compute the probabilities of measuring x = 0 and 1
p_zero = counts.get('01', 0)
p_one = counts.get('11', 0)
total = p_zero + p_one
p_zero /= total
p_one /= total

# The ratio of the coefficints should be 1:9
print(f'Quantum ratio: {p_zero / p_one}')
classical_soln = (np.linalg.inv(A) @ b).reshape(-1)
print(f'Classical solution: {classical_soln}')
print(f'Classical ratio: {classical_soln[0]**2 / classical_soln[1]**2}')

plt.show()
