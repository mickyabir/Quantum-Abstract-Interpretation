import numpy as np

Zero = np.array([[1], [0]], dtype=complex)
One = np.array([[0], [1]], dtype=complex)
Plus = 1 / np.sqrt(2) * np.array([[1], [1]], dtype=complex)
Minus = 1 / np.sqrt(2) * np.array([[1], [-1]], dtype=complex)

def generateTensorState(states):
    state = None
    for s in states:
        if state is None:
            state = s
        else:
            state = np.kron(state, s)

    return state

def generateDensityMatrix(state):
    return np.kron(state, state.conj().T)

def generateDensityMatrixFromQubits(qubits):
    state = generateTensorState(qubits)
    return np.kron(state, state.conj().T)

