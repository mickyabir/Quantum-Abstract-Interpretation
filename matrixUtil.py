import cvxpy as cp
import numpy as np

from unitaryGenerator import generateQubitSwapUnitary, generateQubitRightRotateUnitary

gs_tol = 1e-8
zero_tol = 1e-6
eigval_zero_tol = 1e-10
numpy_round_decimals = 14

def truncateComplexObject(M):
    if type(M) == np.ndarray:
        return np.round(M, numpy_round_decimals)
    # elif type(M) == list:
    #     return [np.round(m, numpy_round_decimals) for m in M]

    return M

# Trace out qubits in F from M, for n qubit system
def partialTrace(M, n, F):
    currentSystemSize = n
    currentMatrix = M

    for i in range(len(F)):
        if F[i] > 0:
            firstSwap = generateQubitSwapUnitary(currentSystemSize, 0, F[i])
            currentMatrix = firstSwap @ currentMatrix @ firstSwap.T

        A = currentMatrix[0:int(currentMatrix.shape[0]/2), 0:int(currentMatrix.shape[1]/2)]
        D = currentMatrix[int(currentMatrix.shape[0]/2):int(currentMatrix.shape[0]), int(currentMatrix.shape[1]/2):int(currentMatrix.shape[1])]

        currentMatrix = A + D

        currentSystemSize -= 1

        F = list(map(lambda x : x - 1, F))

        if F[i] > 0:
            rotateUndo = generateQubitRightRotateUnitary(currentSystemSize, F[i])
            currentMatrix = rotateUndo @ currentMatrix @ rotateUndo.T

    return truncateComplexObject(currentMatrix)

# U is the unitary
# n is the total number of qubits to expand U to
# F is the ordered list of qubits that U applies to
# For now, assume F[i] < F[j] for i < j
def expandUnitary(U, n, F, symbolic=False):
    I = np.identity(2)
    fullUnitary = U

    for i in range(len(F), n):
        if symbolic:
            fullUnitary = cp.kron(fullUnitary, I)
        else:
            fullUnitary = np.kron(fullUnitary, I)

    for i in range(0, len(F)):
        k = len(F) - i - 1
        swapGate = generateQubitSwapUnitary(n, k, F[k])
        fullUnitary = swapGate @ fullUnitary @ swapGate

    if type(fullUnitary) == np.ndarray:
        return np.round(fullUnitary, numpy_round_decimals)

    return truncateComplexObject(fullUnitary)

# from https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def isSemidefinitePositive(A):
    eigvals = np.linalg.eigvals(A)

    eigvals.real[abs(eigvals.real) < eigval_zero_tol] = 0.0
    eigvals.imag[abs(eigvals.imag) < eigval_zero_tol] = 0.0

    return np.all(eigvals >= 0)

def innerProduct(u, v):
    return np.dot(u, v.conj().T)

def vectorProjection(u, v):
    u.real[abs(u.real) < zero_tol] = 0.0
    u.imag[abs(u.imag) < zero_tol] = 0.0

    if not u.any():
        return np.zeros(u.shape[0], dtype=u.dtype)

    u_inner = innerProduct(u, u)
    vu_inner = innerProduct(v, u)
    proj = (vu_inner / u_inner) * u

    return truncateComplexObject(proj)

def gramSchmidt(vectors):
    u_vectors = []
    for i in range(len(vectors)):
        u_i = vectors[i]

        for j in range(0, len(u_vectors)):
            proj = vectorProjection(u_vectors[j], vectors[i])
            u_i = u_i - proj

        u_i.real[abs(u_i.real) < zero_tol] = 0.0
        u_i.imag[abs(u_i.imag) < zero_tol] = 0.0

        if u_i.any():
            u_vectors.append(u_i)

    e_vectors = [truncateComplexObject(u / np.sqrt(innerProduct(u, u))) for u in u_vectors if u.any()]

    return e_vectors


