import numpy as np

gs_tol = 1e-8
zero_tol = 1e-6

def pprint_repr(A):
    retStr = ''
    p_tol = 1e-5
    round_tol = 4
    for i in range(len(A)):
        for j in range(len(A[i])):
            realFlag = False
            complexFlag = False
            if abs(A[i][j].real) > p_tol:
                retStr += f'{np.around(A[i][j].real, decimals = round_tol)}'
                realFlag = True
            if abs(A[i][j].imag) > p_tol:
                imagStr = ''
                if realFlag:
                    imagStr += ' + '

                imagStr += f'{np.around(A[i][j].imag, decimals = round_tol)}j'

                retStr += imagStr

                complexFlag = True

            if not realFlag and not complexFlag:
                retStr += f'0'

            retStr += '   '

        retStr += '\n\n'

    return retStr

def pprint(A):
    print(pprint_repr(A))

class AbstractState():
    def __init__(self, n, S, projections):
        self.n = n
        self.S = S
        self.projections = projections

    def __repr__(self):
        retStr = ""
        retStr += f"Abstract State:\n\nn: {self.n}\n"
        for i in range(len(self.projections)):
            retStr += f"\n{self.S[i]}\n<->\n{pprint_repr(self.projections[i])}\n\n"
        return retStr

def generateQubitRightRotateUnitary(n, i):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    rotateUnitary = np.identity(2 ** n)
    for k in range(2 ** n):
        binaryStringK = list(getBin(k, n))

        leftStringK = binaryStringK[0:i + 1]
        rightStringK = binaryStringK[i + 1:len(binaryStringK)]
        leftStringK = [leftStringK[-1]] + leftStringK[0:len(leftStringK) - 1]
        intK = int(''.join(leftStringK + rightStringK), 2)

        rotateUnitary[k][k] = 0
        rotateUnitary[k][intK] = 1

    return rotateUnitary

def generateQubitLeftRotateUnitary(n, i):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    rotateUnitary = np.identity(2 ** n)
    for k in range(2 ** n):
        binaryStringK = list(getBin(k, n))

        leftStringK = binaryStringK[0:i + 1]
        rightStringK = binaryStringK[i + 1:len(binaryStringK)]
        leftStringK = leftStringK[1:len(leftStringK)] + [leftStringK[0]]
        intK = int(''.join(leftStringK + rightStringK), 2)

        rotateUnitary[k][k] = 0
        rotateUnitary[k][intK] = 1

    return rotateUnitary

def generateQubitSwapUnitary(n, i, j):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    swapUnitary = np.identity(2 ** n)

    for k in range(2 ** n):
        binaryStringK = list(getBin(k, n))
        tmp = binaryStringK[i]
        binaryStringK[i] = binaryStringK[j]
        binaryStringK[j] = tmp
        binaryStringK = ''.join(binaryStringK)
        intK = int(binaryStringK, 2)

        swapUnitary[k][k] = 0
        swapUnitary[k][intK] = 1

    return swapUnitary


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

def intersectProjections(projections, tj):
    expandedUnion = None
    for p in projections:
        if expandedUnion is None:
            expandedUnion = p.copy()
        else:
            expandedUnion += p

    fullComplementUnion = len(projections) * np.identity(2 ** len(tj)) - expandedUnion
    complementSupport = getSupport(fullComplementUnion)

    if not complementSupport:
        return np.identity(2 ** len(tj), dtype=complex)
    complementSupportMatrix = getMatrixFromSpan(complementSupport)

    finalSupportMatrix = np.identity(2 ** len(tj)) - complementSupportMatrix

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
        T.append(sorted(list(set(si).union(set(F)))))

    concreteState = gammaFunction(state, T)
    evolvedState = applyGate(concreteState, U, F)
    evolvedAbstractState = alphaFunction(evolvedState, state.S)

    return evolvedAbstractState

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

def getSupport(A):
    columnVectors = [A[:, i] for i in range(A.shape[0])]
    gsColumnVectors = gramSchmidt(columnVectors)
    return gsColumnVectors

def getMatrixFromSpan(span):
    dim = span[0].shape[0]
    P_span = np.zeros((dim, dim), dtype=complex)

    for i in range(len(span)):
        P_span[:, i] = span[i]

    return P_span @ P_span.conj().T

def generateGHZPaperFull(n):
    S = []

    for i in range(n):
        for j in range(i + 1, n):
            S.append([i, j])

    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(n, S, [initial_proj for _ in range(int(n * (n - 1) / 2))])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = initial_state

    for i in range(0, n - 1):
        nextState = abstractStep(nextState, H, [i])

    nextState = abstractStep(nextState, X, [n - 1])

    for i in range(0, n - 1):
        nextState = abstractStep(nextState, CNOT, [i, n - 1])

    for i in range(0, n):
        nextState = abstractStep(nextState, H, [i])

def generateGHZPaperPartial(n):
    S = []

    for i in range(n - 1):
        S.append([i, i + 1])

    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(n, S, [initial_proj for _ in range(n - 1)])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = initial_state

    for i in range(0, n - 1):
        nextState = abstractStep(nextState, H, [i])

    nextState = abstractStep(nextState, X, [n - 1])

    for i in range(0, n - 1):
        nextState = abstractStep(nextState, CNOT, [i, n - 1])

    for i in range(0, n):
        nextState = abstractStep(nextState, H, [i])

def generateGHZFull(n):
    S = []

    for i in range(n):
        for j in range(i + 1, n):
            S.append([i, j])

    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(n, S, [initial_proj for _ in range(int(n * (n - 1) / 2))])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = abstractStep(initial_state, H, [0])


    for i in range(1, n):
        nextState = abstractStep(nextState, CNOT, [0, i])

def exampleFromPaper():
    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(3, [[0, 1], [0, 2], [1, 2]], [initial_proj, initial_proj, initial_proj])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    print(initial_state)

    state1 = abstractStep(initial_state, H, [0])
    print(state1)

    state2 = abstractStep(state1, H, [1])
    print(state2)

    state3 = abstractStep(state2, X, [2])
    print(state3)

    state4 = abstractStep(state3, CNOT, [1, 2])
    print(state4)

    state5 = abstractStep(state4, CNOT, [0, 2])
    print(state5)

    state6 = abstractStep(state5, H, [0])
    print(state6)

    state7 = abstractStep(state6, H, [1])
    print(state7)

    state8 = abstractStep(state7, H, [2])
    print(state8)

if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    generateGHZPaperPartial(3)
    0/0

    # exampleFromPaper()

    # import time

    # qubitList = [3, 5, 10, 15, 20, 30, 40, 50]

    # for n in qubitList:
    #     prev = time.time()
    #     generateGHZPaperFull(n)
    #     elapsed = time.time() - prev
    #     print(f'{n}: {elapsed}')

    import cProfile, pstats, io
    from pstats import SortKey

    with cProfile.Profile() as pr:
        generateGHZPaperFull(20)

        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

