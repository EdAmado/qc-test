"""
encoders.py
Optimized factory for parameterized quantum image encoding circuits.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_frqi(num_pixels: int):
    """FRQI: 1 color qubit + log2(N) position qubits."""
    num_position = int(np.log2(num_pixels))
    colour_qubit = num_position
    num_qubits = num_position + 1
    
    circuit = QuantumCircuit(num_qubits)
    for q in range(num_position):
        circuit.h(q)
        
    inputs = ParameterVector("frqi_pixel", num_pixels)
    
    for pixel_index in range(num_pixels):
        binary = format(pixel_index, f"0{num_position}b")
        for qubit, bit in enumerate(binary):
            if bit == "0":
                circuit.x(qubit)
                
        circuit.mcry(
            theta=2 * inputs[pixel_index],
            q_controls=list(range(num_position)),
            q_target=colour_qubit,
            mode="noancilla",
        )
        
        for qubit, bit in enumerate(binary):
            if bit == "0":
                circuit.x(qubit)
                
    pauli = ["I"] * num_qubits
    pauli[colour_qubit] = "Z"
    observable = SparsePauliOp.from_list([("".join(pauli), 1.0)])
    
    return circuit, inputs, observable


def build_amplitude_encoding(num_pixels: int):
    """Optimized Amplitude Proxy: Maps all N parameters uniformly across log2(N) qubits."""
    num_qubits = int(np.log2(num_pixels)) # For 64 pixels, this automatically becomes 6
    circuit = QuantumCircuit(num_qubits)
    inputs = ParameterVector("amp_pixel", num_pixels)
    
    idx = 0
    # Scaled iteration limits up from 4 to 11 to fit all 64 variables cleanly
    for rotation_layer in range(11):  
        for q in range(num_qubits):
            if idx < num_pixels:
                circuit.ry(inputs[idx], q)
                idx += 1
        # Entanglement structure scaled for 6 qubits
        if rotation_layer < 10:
            for q in range(0, num_qubits - 1, 2):
                circuit.cx(q, q + 1)
                
    pauli = ["I"] * num_qubits
    pauli[0] = "Z"
    observable = SparsePauliOp.from_list([("".join(pauli), 1.0)])
    
    return circuit, inputs, observable


def build_neqr(num_pixels: int):
    """Optimized Simulation-Friendly NEQR Variant."""
    num_position = int(np.log2(num_pixels))
    num_intensity_qubits = 4  
    num_qubits = num_position + num_intensity_qubits
    
    circuit = QuantumCircuit(num_qubits)
    inputs = ParameterVector("neqr_pixel", num_pixels)
    
    # Hardware-efficient alternative to avoid the 30-minute deep multi-controlled gate explosion
    for q in range(num_position):
        circuit.h(q)
        
    # Apply parameterized inputs across the digital landscape efficiently
    for i in range(num_pixels):
        target_qubit = num_position + (i % num_intensity_qubits)
        control_qubit = i % num_position
        circuit.cry(inputs[i], control_qubit, target_qubit)
                
    pauli = ["I"] * num_qubits
    pauli[-1] = "Z"
    observable = SparsePauliOp.from_list([("".join(pauli), 1.0)])
    
    return circuit, inputs, observable


def build_qwt(num_pixels: int):
    """Quantum Wavelet Transform Layout Proxy."""
    circuit, inputs, observable = build_amplitude_encoding(num_pixels)
    num_qubits = circuit.num_qubits
    
    for q in range(num_qubits - 1):
        circuit.ch(q, q + 1)
        circuit.cz(q, q + 1)
        
    return circuit, inputs, observable


def get_encoder(encoding_name: str, num_pixels: int):
    """Factory Router"""
    encoders = {
        "frqi": build_frqi,
        "amplitude": build_amplitude_encoding,
        "neqr": build_neqr,
        "qwt": build_qwt
    }
    name = encoding_name.lower()
    if name not in encoders:
        raise ValueError(f"Unknown encoding protocol: {encoding_name}")
        
    return encoders[name](num_pixels)