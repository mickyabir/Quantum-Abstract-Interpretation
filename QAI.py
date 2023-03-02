import numpy as np

tol = 1e-14

class AbstractState():
    def __init__(self, n, S, projections):
        self.n = n
        self.S = S
        self.projections = projections

    def __repr__(self):
        retStr = ""
        retStr += f"Abstract State:\n\nn: {self.n}\n"
        for i in range(len(self.projections)):
            retStr += f"\n{self.S[i]}\n<->\n{self.projections[i]}\n\n"
        return retStr

def generateQubitRightRotateUnitary(n, i):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    ketMap = {'0': np.array([1, 0]), '1': np.array([0, 1])}

    rotateUnitary = None
    for k in range(2 ** n):
        binaryStringK = getBin(k, n)
        rotateUnitaryRow = None
        for l in range(len(binaryStringK)):
            elem = None

            if l > i:
                elem = binaryStringK[l]
            else: 
                elem = binaryStringK[(l + i) % (i + 1)]

            if rotateUnitaryRow is None:
                rotateUnitaryRow = ketMap[elem]
            else:
                rotateUnitaryRow = np.kron(rotateUnitaryRow, ketMap[elem])

        if rotateUnitary is None:
            rotateUnitary = rotateUnitaryRow
        else:
            rotateUnitary = np.vstack((rotateUnitary, rotateUnitaryRow))

    return rotateUnitary

def generateQubitLeftRotateUnitary(n, i):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    ketMap = {'0': np.array([1, 0]), '1': np.array([0, 1])}

    rotateUnitary = None
    for k in range(2 ** n):
        binaryStringK = getBin(k, n)
        rotateUnitaryRow = None
        for l in range(len(binaryStringK)):
            elem = None

            if l > i:
                elem = binaryStringK[l]
            else: 
                elem = binaryStringK[(l - i) % (i + 1)]

            if rotateUnitaryRow is None:
                rotateUnitaryRow = ketMap[elem]
            else:
                rotateUnitaryRow = np.kron(rotateUnitaryRow, ketMap[elem])

        if rotateUnitary is None:
            rotateUnitary = rotateUnitaryRow
        else:
            rotateUnitary = np.vstack((rotateUnitary, rotateUnitaryRow))

    return rotateUnitary


def generateQubitSwapUnitary(n, i, j):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    ketMap = {'0': np.array([1, 0]), '1': np.array([0, 1])}

    swapUnitary = None
    for k in range(2 ** n):
        binaryStringK = getBin(k, n)
        swapUnitaryRow = None
        for l in range(len(binaryStringK)):
            elem = None

            if l == i:
                elem = binaryStringK[j]
            elif l == j:
                elem = binaryStringK[i]
            else:
                elem = binaryStringK[l]

            if swapUnitaryRow is None:
                swapUnitaryRow = ketMap[elem]
            else:
                swapUnitaryRow = np.kron(swapUnitaryRow, ketMap[elem])

        if swapUnitary is None:
            swapUnitary = swapUnitaryRow
        else:
            swapUnitary = np.vstack((swapUnitary, swapUnitaryRow))

    return swapUnitary

def generateQubitSwapFrontUnitary(n, F):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    ketMap = {'0': np.array([1, 0]), '1': np.array([0, 1])}

    indexMap = {F[i]:i for i in range(len(F))}
    indexMapReverse = {i:F[i] for i in range(len(F))}
    indexMap.update(indexMapReverse)

    swapUnitary = None
    for k in range(2 ** n):
        binaryStringK = getBin(k, n)
        swapUnitaryRow = None
        for l in range(len(binaryStringK)):
            if l in indexMap.keys():
                elem = binaryStringK[indexMap[l]]
            else:
                elem = binaryStringK[l]

            if swapUnitaryRow is None:
                swapUnitaryRow = ketMap[elem]
            else:
                swapUnitaryRow = np.kron(swapUnitaryRow, ketMap[elem])

        if swapUnitary is None:
            swapUnitary = swapUnitaryRow
        else:
            swapUnitary = np.vstack((swapUnitary, swapUnitaryRow))

    return swapUnitary

# Trace out qubits in F from M, for n qubit system
def partialTrace(M, n, F):
    currentSystemSize = n
    currentMatrix = M

    for i in F:
        firstSwap = generateQubitSwapUnitary(currentSystemSize, 0, i)

        currentMatrix = firstSwap @ currentMatrix

        A = currentMatrix[0:int(M.shape[0]/2), 0:int(M.shape[1]/2)]
        D = currentMatrix[int(M.shape[0]/2):int(M.shape[0]), int(M.shape[1]/2):int(M.shape[1])]

        currentMatrix = A + D

        currentSystemSize -= 1
        rotateUndo = generateQubitRightRotateUnitary(currentSystemSize, i - 1)
        currentMatrix = rotateUndo @ currentMatrix

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

def intersectProjections(projections, tj):
    expandedUnion = None
    for p in projections:
        if expandedUnion is None:
            expandedUnion = p.copy()
        else:
            expandedUnion += p

    fullComplementUnion = len(projections) * np.identity(2 ** len(tj)) - expandedUnion
    complementSupport = getSupport(fullComplementUnion)
    complementSupportMatrix = getMatrixFromSpan(complementSupport)

    complementSupportMatrix.real[abs(complementSupportMatrix.real) < tol] = 0.0
    complementSupportMatrix.imag[abs(complementSupportMatrix.imag) < tol] = 0.0

    finalSupportMatrix = np.identity(2 ** len(tj)) - complementSupportMatrix
    finalSupportMatrix.real[abs(finalSupportMatrix.real) < tol] = 0.0
    finalSupportMatrix.imag[abs(finalSupportMatrix.imag) < tol] = 0.0

    return finalSupportMatrix

def gammaFunction(state, T):
    Q = []
    m = len(state.S)
    assert(m == len(T))

    for j in range(m):
        projectionIntersectionList = []

        forwardMap = {T[j][i]:i for i in range(len(T[j]))}
        applyForwardMap = lambda S: [forwardMap[si] for si in S]

        for i in range(m):
            if set(state.S[i]).issubset(set(T[j])):
                mappedS = applyForwardMap(state.S[i])
                expandedProjection = expandUnitary(state.projections[i], len(T[j]), mappedS)
                projectionIntersectionList.append(expandedProjection)

        Q.append(intersectProjections(projectionIntersectionList, T[j]))

    return AbstractState(state.n, T, Q)

def applyGate(state, U, F):
    applyGateToProjection = lambda U, p: U @ p @ U.conj().T
    evolvedProjections = []
    for i in range(len(state.S)):
        forwardMap = {state.S[i][j]:j for j in range(len(state.S[i]))}
        applyForwardMap = lambda S: [forwardMap[si] for si in S]
        mappedF = applyForwardMap(F)

        expandedU = expandUnitary(U, len(state.S[i]), mappedF)
        evolvedExpandedU = applyGateToProjection(expandedU, state.projections[i])
        evolvedExpandedU.real[abs(evolvedExpandedU.real) < tol] = 0.0
        evolvedExpandedU.imag[abs(evolvedExpandedU.imag) < tol] = 0.0
        evolvedProjections.append(evolvedExpandedU)
    return AbstractState(state.n, state.S, evolvedProjections)

def alphaFunction(state, S):
    P = []
    m = len(state.S)
    assert(m == len(S))

    for i in range(m):
        subsystemIntersectionList = []

        for j in range(m):
            forwardMap = {state.S[j][i]:i for i in range(len(state.S[j]))}
            applyForwardMap = lambda S: [forwardMap[si] for si in S]

            if set(S[i]).issubset(set(state.S[j])):
                mappedS = applyForwardMap(S[i])
                mappedT = applyForwardMap(state.S[j])
                traceoutQubits = list(set(mappedT).difference(set(mappedS)))
                subsystem = partialTrace(state.projections[j], len(state.S[j]), traceoutQubits)
                subsystemSupportSpan = getSupport(subsystem)
                subsystemSupport = getMatrixFromSpan(subsystemSupportSpan)
                subsystemIntersectionList.append(subsystemSupport)

        P.append(intersectProjections(subsystemIntersectionList, S[i]))

    return AbstractState(state.n, S, P)

def abstractStep(state, U, F):
    T = []
    for si in state.S:
        T.append(list(set(si).union(set(F))))

    print('######################################\n\n')
    concreteState = gammaFunction(state, T)
    print(f'Concrete State:\n{concreteState}')

    evolvedState = applyGate(concreteState, U, F)
    print(f'Evolved Concrete State:\n{evolvedState}')

    evolvedAbstractState = alphaFunction(evolvedState, state.S)
    print(f'Evolved Abstract State:\n{evolvedAbstractState}')
    print('######################################\n\n')
    return evolvedAbstractState

# from https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def vecotrProjection(u, v):
    v_norm = np.sqrt(np.dot(v.T, v).item())
    proj = (np.dot(u.T, v).item()/v_norm**2)*v
    return proj

def gramSchmidt(vectors):
    u_vectors = []
    for i in range(len(vectors)):
        u_i = vectors[i]

        for j in range(0, i):
            proj = vecotrProjection(vectors[i], vectors[j])
            u_i = u_i - proj
            u_i.real[abs(u_i.real) < tol] = 0.0
            u_i.imag[abs(u_i.imag) < tol] = 0.0

        if u_i.any():
            u_vectors.append(u_i)

    e_vectors = [u / np.sqrt(np.dot(u.T, u).item()) for u in u_vectors]

    return e_vectors

def getSupport(A):
    w, v = np.linalg.eig(A)

    vectors = []
    for i in range(len(w)):
        if w[i] > 0:
            vector = v[:, i]
            vector.real[abs(vector.real) < tol] = 0.0
            vector.imag[abs(vector.imag) < tol] = 0.0
            vectors.append(vector)

    return vectors

def getMatrixFromSpan(span):
    dim = span[0].shape[0]
    # P_span = np.array(np.zeros((dim, dim)), dtype=complex)
    P_span = np.zeros((dim, dim), dtype=complex)

    # Gram Schmidt
    # TODO: Might be not necessary
    orthonorm_span = gramSchmidt(span)

    for i in range(len(orthonorm_span)):
        P_span[:, i] = orthonorm_span[i]

    return P_span @ P_span.conj().T

def intersectSupports(suppA, suppB):
    dim = suppA[0].shape[0]
    I = np.identity(dim)

    P_A = getMatrixFromSpan(suppA)
    P_B = getMatrixFromSpan(suppB)

    P_A_comp = I - P_A
    P_B_comp = I - P_B

    supp_comp_union = getSupport(P_A_comp + P_B_comp)

    P_comp_union = getMatrixFromSpan(supp_comp_union)

    P_intersect = I - P_comp_union

    P_intersect.real[abs(P_intersect.real) < tol] = 0.0
    P_intersect.imag[abs(P_intersect.imag) < tol] = 0.0

    return P_intersect

if __name__ == '__main__':
    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(3, [[0, 1], [0, 2], [1, 2]], [initial_proj, initial_proj, initial_proj])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)

    print(initial_state)

    state1 = abstractStep(initial_state, H, [0])
    print(state1)

    import pdb
    pdb.set_trace()

    state2 = abstractStep(state1, H, [1])
    print(state2)

    state3 = abstractStep(state2, X, [2])
    print(state3)
