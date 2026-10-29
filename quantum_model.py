"""
quantum_model.py (Refactored for Modular Encodings)
"""

import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from encoders import get_encoder

class DynamicHybridQNN(nn.Module):
    def __init__(self, image_size: int = 4, reps: int = 2, encoding: str = "frqi"):
        super().__init__()
        self.num_pixels = image_size * image_size
        
        # 1. Dynamically fetch selected encoding circuit, input vector, and custom observable
        frqi_circ, input_params, observable = get_encoder(encoding, self.num_pixels)
        num_qubits = frqi_circ.num_qubits
        
        # 2. Build Variational Ansatz
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement="linear")
        
        # 3. Combine Circuit pipeline
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(frqi_circ, inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuit = circuit.decompose()
        
        # 4. Engine Accelerator Setup
        aer_backend = AerSimulator(method="statevector", max_parallel_threads=0)
        aer_estimator = BackendEstimatorV2(backend=aer_backend)
        
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=input_params,
            weight_params=ansatz.parameters,
            estimator=aer_estimator, 
        )
        
        # 5. Connect to PyTorch Engine
        self.quantum = TorchConnector(qnn)
        
    def forward(self, x):
        return self.quantum(x)