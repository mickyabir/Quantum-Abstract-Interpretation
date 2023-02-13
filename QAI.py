import logging

import numpy as np

tol = 1e-16

class AbstractState():
    def __init__(self, n, S, projections):
        self.n = n
        self.S = S
        self.projections = projections

def generateQubitSwapUnitary(n, i, j):
    if n == 1:
        return np.identity(2)

    get_bin = lambda x, n: format(x, 'b').zfill(n)

    ketMap = {'0': np.matrix([[1, 0]]), '1': np.matrix([[0, 1]])}

    swapUnitary = None
    for k in range(2 ** n):
        binaryStringK = get_bin(k, n)
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

# Trace out qubits in F from M, for n qubit system
def partialTrace(M, n, F):
    currentSystemSize = n
    currentMatrix = M

    for i in F:
        firstSwap = generateQubitSwapUnitary(currentSystemSize, 0, i)

        currentMatrix = np.matmul(currentMatrix, firstSwap)

        A = M[0:int(M.shape[0]/2), 0:int(M.shape[1]/2)]
        D = M[int(M.shape[0]/2):int(M.shape[0]), int(M.shape[1]/2):int(M.shape[1])]

        currentMatrix = A + D

        currentSystemSize -= 1
        secondSwap = generateQubitSwapUnitary(currentSystemSize, 0, i)
        currentMatrix = np.matmul(currentMatrix, secondSwap)

    return currentMatrix

# U is the unitary
# n is the total number of qubits to expand U to
# F is the ordered list of qubits that U applies to
# For now, assume F[i] < F[j] for i < j
def expandUnitary(U, n, F):
    fullSwapUnitary = None
    for i in range(len(F)):
        swapGate = generateQubitSwapUnitary(n, i, F[i])
        if fullSwapUnitary is None:
            fullSwapUnitary = swapGate
        else:
            fullSwapUnitary = np.matmul(fullSwapUnitary, swapGate)

    I = np.identity(2)
    fullUnitary = U
    for i in range(len(F), n):
        fullUnitary = np.kron(fullUnitary, I)

    fullUnitaryLeftSwap = np.matmul(fullSwapUnitary, fullUnitary)
    fullUnitary = np.matmul(fullUnitary, fullSwapUnitary)
    return fullUnitary

def intersectProjections(projections, tj):
    expandedUnion = None
    for p in projections:
        if expandedUnion is None:
            expandedUnion = p
        else:
            expandedUnion += p

    fullComplementUnion = len(projections) * np.identity(2 ** len(tj)) - expandedUnion
    complementSupport = get_support(fullComplementUnion)
    complementSupportMatrix = get_matrix_from_span(complementSupport)

    return np.identity(2 ** len(tj)) - complementSupportMatrix


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
    applyGateToProjection = lambda U, p: np.matmul(U, np.matmul(p, U.conj().T))
    evolvedProjections = []
    for i in range(len(state.S)):
        forwardMap = {state.S[i][j]:j for j in range(len(state.S[i]))}
        applyForwardMap = lambda S: [forwardMap[si] for si in S]
        mappedF = applyForwardMap(F)

        expandedU = expandUnitary(U, len(state.S[i]), mappedF)
        evolvedProjections.append(applyGateToProjection(expandedU, state.projections[i]))
    return AbstractState(state.n, state.S, evolvedProjections)

def abstractStep(state, U, F):
    T = []
    for si in state.S:
        T.append(list(set(si).union(set(F))))

    concreteState = gammaFunction(state, T)

    import pdb
    pdb.set_trace()
    evolvedState = applyGate(concreteState, U, F)

# from https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def vector_projection(u, v):
    v_norm = np.sqrt(np.dot(v.T, v).item())
    proj = (np.dot(u.T, v).item()/v_norm**2)*v
    return proj

def gram_schmidt(vectors):
    u_vectors = []
    for i in range(len(vectors)):
        u_i = vectors[i]

        for j in range(0, i):
            proj = vector_projection(vectors[i], vectors[j])
            u_i = u_i - proj
            # u_i.real[abs(u_i.real) < tol] = 0.0
            # u_i.imag[abs(u_i.imag) < tol] = 0.0
            u_i[abs(u_i) < tol] = 0.0

        u_vectors.append(u_i)

    e_vectors = [u / np.sqrt(np.dot(u.T, u).item()) for u in u_vectors]

    return e_vectors

def get_support(A):
    w, v = np.linalg.eig(A)

    vectors = []
    for i in range(len(w)):
        if w[i] > 0:
            # vectors.append(normalized(v[:, i]) / np.sqrt(w[i]))
            # vectors.append(normalized(v[:, i]))
            vectors.append(v[:, i])

    return vectors

def get_matrix_from_span(span):
    dim = span[0].shape[0]
    P_span = np.matrix(np.zeros((dim, dim)))

    # Gram Schmidt
    # TODO: Might be not necessary
    orthonorm_span = gram_schmidt(span)

    for i in range(len(span)):
        P_span[:, i] = orthonorm_span[i]

    return P_span * P_span.conj().T

def intersect_supports(suppA, suppB):
    dim = suppA[0].shape[0]
    I = np.identity(dim)

    P_A = get_matrix_from_span(suppA)
    P_B = get_matrix_from_span(suppB)

    logging.debug(f"\nP_A:\n{P_A}\n")
    logging.debug(f"\nP_B:\n{P_B}\n")

    P_A_comp = I - P_A
    P_B_comp = I - P_B

    logging.debug(f"\nP_A_comp:\n{P_A_comp}\n")
    logging.debug(f"\nP_B_comp:\n{P_B_comp}\n")
    logging.debug(f"\nP_add:\n{P_A_comp + P_B_comp}\n")

    supp_comp_union = get_support(P_A_comp + P_B_comp)

    logging.debug(f"\nsupp_comp_union:\n{supp_comp_union}\n")

    P_comp_union = get_matrix_from_span(supp_comp_union)
    P_comp_union[abs(P_comp_union) < tol] = 0.0

    logging.debug(f"\nP_comp_union:\n{P_comp_union}\n")

    P_intersect = I - P_comp_union

    #TODO: Check that this works
    P_intersect[abs(P_intersect) < tol] = 0.0

    logging.debug(f"\nP_intersect:\n{P_intersect}\n")

    return P_intersect

if __name__ == '__main__':
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # A = np.matrix([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    # B = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    # logging.debug(f'\nA:\n{A}\n')
    # logging.debug(f'\nA:\n{B}\n')
    # suppA = get_support(A)
    # suppB = get_support(B)
    # logging.debug(f'\nsuppA:\n{suppA}\n')
    # logging.debug(f'\nsuppB:\n{suppB}\n')

    # P_intersect = intersect_supports(suppA, suppB)
    # P_intersect_support = get_support(P_intersect)
    # logging.debug(f'\nP_intersect_support:\n{P_intersect_support}\n')
    # U = np.matrix([[0, 1],[1, 0]])
    # n = 3
    # F = [1]
    # U = np.matrix([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    # n = 3
    # F = [0, 1]
    # fullUnitary = expandUnitary(U, n, F)
    # print(fullUnitary)

    # M = np.matrix([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    # n = 2
    # F = [0]
    # p = partialTrace(M, n, F)
    # print(p)

    initial_proj = np.matrix([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    initial_state = AbstractState(3, [[0, 1], [0, 2], [1, 2]], [initial_proj, initial_proj, initial_proj])

    H = 1/np.sqrt(2) * np.matrix([[1, 1],[1, -1]])
    state1 = abstractStep(initial_state, H, [0])


