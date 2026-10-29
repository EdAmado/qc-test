"""
frqi.py
Flexible Representation of Quantum Images (FRQI)
"""

from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit


# Image preprocessing

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize grayscale image into [0,1].
    """
    return image.astype(np.float64) / 255.0


def pixel_to_angle(pixel: float) -> float:
    """
    Convert a grayscale value into an FRQI angle.
    """
    return (pixel / 255.0) * (np.pi / 2)


def compute_angles(image: np.ndarray) -> np.ndarray:
    """
    Compute one rotation angle per pixel.
    """
    return np.array(
        [pixel_to_angle(p) for p in image.flatten()]
    )


# Quantum helpers

def create_position_superposition(
    circuit: QuantumCircuit,
    num_position_qubits: int,
):
    """
    Put every position qubit into superposition.
    """

    for q in range(num_position_qubits):
        circuit.h(q)


def apply_pixel_rotation(
    circuit: QuantumCircuit,
    pixel_index: int,
    angle: float,
    num_position_qubits: int,
    colour_qubit: int,
):
    """
    Encode one pixel using a multi-controlled RY.
    """

    binary = format(
        pixel_index,
        f"0{num_position_qubits}b"
    )

    #
    # Controls must correspond to |111...>
    #
    # Therefore invert qubits where the
    # binary digit is zero.
    #

    for qubit, bit in enumerate(binary):

        if bit == "0":
            circuit.x(qubit)

    controls = list(range(num_position_qubits))

    circuit.mcry(
        theta=2 * angle,
        q_controls=controls,
        q_target=colour_qubit,
        q_ancillae=[],
        mode="noancilla",
    )

    # Restore state

    for qubit, bit in enumerate(binary):

        if bit == "0":
            circuit.x(qubit)


# Main encoder

def frqi_encode(
    image: np.ndarray,
) -> QuantumCircuit:
    """
    Encode a grayscale image using FRQI.

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    QuantumCircuit
    """

    pixels = image.flatten()
    num_pixels = len(pixels)

    if not np.log2(num_pixels).is_integer():
        raise ValueError(
            "Image must contain a power-of-two number of pixels."
        )

    num_position = int(np.log2(num_pixels))
    colour_qubit = num_position
    circuit = QuantumCircuit(num_position + 1)

    create_position_superposition(
        circuit,
        num_position,
    )

    angles = compute_angles(image)

    for pixel_index, angle in enumerate(angles):
        apply_pixel_rotation(
            circuit,
            pixel_index,
            angle,
            num_position,
            colour_qubit,
        )

    return circuit


# Utilities

def draw_frqi_circuit(
    circuit: QuantumCircuit,
    output="mpl",
):
    return circuit.draw(output)


def get_frqi_resources(
    circuit: QuantumCircuit,
):

    return {
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "parameters": circuit.num_parameters,
        "operations": circuit.count_ops(),
    }


# Example

if __name__ == "__main__":

    image = np.random.randint(
        0,
        255,
        (4, 4),
        dtype=np.uint8,
    )

    qc = frqi_encode(image)
    print(get_frqi_resources(qc))
    print(qc.draw())