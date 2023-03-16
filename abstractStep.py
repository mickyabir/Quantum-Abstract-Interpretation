from abstractState import AbstractState
from matrixUtil import expandUnitary, partialTrace
from projections import intersectProjections, getSupport, getMatrixFromSpan

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


