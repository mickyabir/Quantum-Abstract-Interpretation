import numpy as np


H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
T = np.array([[1, 0],[0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0],[0, np.exp(-1j * np.pi / 4)]], dtype=complex)
X = np.array([[0, 1],[1, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CNOT10 = np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)

def generatePhaseGate(m, inverse=False):
    if inverse:
        return np.array([[1, 0], [0, np.exp(-2 * np.pi * 1j / (2 ** m))]], dtype=complex)
    else:
        return np.array([[1, 0], [0, np.exp(2 * np.pi * 1j / (2 ** m))]], dtype=complex)

def generateControlPhaseGate(m, inverse=False):
    phaseGate = generatePhaseGate(m, inverse)

    zeroZero = np.array([[1, 0], [0, 0]], dtype=complex)
    oneOne = np.array([[0, 0], [0, 1]], dtype=complex)
    identity = np.identity(2)
    return np.kron(zeroZero, identity) + np.kron(oneOne, phaseGate)


