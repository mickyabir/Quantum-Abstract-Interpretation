import numpy as np

H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
T = np.array([[1, 0],[0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0],[0, np.exp(-1j * np.pi / 4)]], dtype=complex)
X = np.array([[0, 1],[1, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CNOT10 = np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

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

def generateControlUGate(U):
    zeroZero = np.array([[1, 0], [0, 0]], dtype=complex)
    oneOne = np.array([[0, 0], [0, 1]], dtype=complex)
    identity = np.identity(U.shape[0])
    return np.kron(zeroZero, identity) + np.kron(oneOne, U)

def generateRandomSpecialUnitary2(eps):
    alpha = np.random.rand() * (2 * np.pi) * eps
    phi = np.random.rand() * (np.pi / 2) * eps
    psi = np.random.rand() * (2 * np.pi) * eps
    chi = np.random.rand() * (2 * np.pi) * eps

    a_11 = np.exp(1j * psi) * np.cos(phi)
    a_12 = np.exp(1j * chi) * np.sin(phi)
    a_21 = -np.exp(-1j * psi) * np.sin(phi)
    a_22 = np.exp(-1j * psi) * np.cos(phi)
    return np.exp(1j * alpha) * np.array([[a_11, a_12],[a_21, a_22]], dtype=complex)

def generateNaiveNoise(n, eps):
    if n == 1:
        return generateRandomSpecialUnitary2(eps)
    elif n == 2:
        return np.kron(generateRandomSpecialUnitary2(eps), generateRandomSpecialUnitary2(eps))
    else:
        raise NotImplementedError

def generateNaiveNoisyGate(gate, eps):
    if gate.shape[0] == 2:
        noise = generateRandomSpecialUnitary2(eps)
    elif gate.shape[0] == 4:
        noise = np.kron(generateRandomSpecialUnitary2(eps), generateRandomSpecialUnitary2(eps))
    else:
        raise NotImplementedError
    return noise @ gate @ noise.conj().T
