import numpy as np

from unitaryGenerator import generateQubitSwapUnitary, generateQubitRightRotateUnitary

gs_tol = 1e-8
zero_tol = 1e-6

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

    return currentMatrix

# U is the unitary
# n is the total number of qubits to expand U to
# F is the ordered list of qubits that U applies to
# For now, assume F[i] < F[j] for i < j
def expandUnitary(U, n, F):
    I = np.identity(2)
    fullUnitary = U

    for i in range(len(F), n):
        fullUnitary = np.kron(fullUnitary, I)

    for i in range(0, len(F)):
        k = len(F) - i - 1
        swapGate = generateQubitSwapUnitary(n, k, F[k])
        fullUnitary = swapGate @ fullUnitary @ swapGate

    return fullUnitary

# from https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def vectorProjection(u, v):
    v_norm = np.sqrt(np.dot(v.T, v).item())
    proj = (np.dot(u.T, v).item() / v_norm ** 2) * v
    return proj

def gramSchmidt(vectors):
    u_vectors = []
    for i in range(len(vectors)):
        u_i = vectors[i]

        for j in range(0, len(u_vectors)):
            proj = vectorProjection(vectors[i], u_vectors[j])
            u_i = u_i - proj

        u_i.real[abs(u_i.real) < zero_tol] = 0.0
        u_i.imag[abs(u_i.imag) < zero_tol] = 0.0

        if u_i.any():
            u_vectors.append(u_i)

    e_vectors = [u / np.sqrt(np.dot(u.T, u).item()) for u in u_vectors]

    return e_vectors


