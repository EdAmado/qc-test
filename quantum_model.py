"""
quantum_model.py (Refactored for Dynamic Inputs)
"""

import torch
import torch.nn as nn
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

def build_param_frqi(num_pixels: int) -> tuple[QuantumCircuit, ParameterVector]:
    """Creates a parameterized FRQI circuit where pixel values are input parameters."""
    num_position = int(np.log2(num_pixels))
    colour_qubit = num_position
    
    circuit = QuantumCircuit(num_position + 1)
    
    # Superposition for positions
    for q in range(num_position):
        circuit.h(q)
        
    # Input parameters for the angles
    inputs = ParameterVector("pixel_angle", num_pixels)
    
    # Controlled rotations for each pixel position
    for pixel_index in range(num_pixels):
        binary = format(pixel_index, f"0{num_position}b")
        
        # Invert qubits where binary digit is '0'
        for qubit, bit in enumerate(binary):
            if bit == "0":
                circuit.x(qubit)
                
        controls = list(range(num_position))
        
        # Pass the parameter placeholder instead of a static float
        circuit.mcry(
            theta=2 * inputs[pixel_index],
            q_controls=controls,
            q_target=colour_qubit,
            mode="noancilla",
        )
        
        # Restore state
        for qubit, bit in enumerate(binary):
            if bit == "0":
                circuit.x(qubit)
                
    return circuit, inputs

class DynamicHybridQNN(nn.Module):
    def __init__(self, image_size: int = 4, reps: int = 2):
        super().__init__()
        self.num_pixels = image_size * image_size
        
        # 1. Build parameterized FRQI Base
        frqi_circ, input_params = build_param_frqi(self.num_pixels)
        num_qubits = frqi_circ.num_qubits
        
        # 2. Build Ansatz
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement="linear")
        
        # 3. Combine Circuits
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(frqi_circ, inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuit = circuit.decompose()
        
        # 4. Define Observable (Z on colour qubit)
        pauli = ["I"] * num_qubits
        pauli[-1] = "Z"
        observable = SparsePauliOp.from_list([("".join(pauli), 1.0)])
        
        # 5. Fixed Accelerated Estimator Setup
        # Use AerSimulator wrapper through BackendEstimator to prevent the V1/V2 internal crash
        aer_backend = AerSimulator(method="statevector", max_parallel_threads=0)
        aer_estimator = BackendEstimatorV2(backend=aer_backend)
        
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=input_params,
            weight_params=ansatz.parameters,
            estimator=aer_estimator, 
        )
        
        # 6. Connect to PyTorch
        self.quantum = TorchConnector(qnn)
        
    def forward(self, x):
        # x shape expected: [batch_size, num_pixels] containing FRQI angles
        return self.quantum(x)