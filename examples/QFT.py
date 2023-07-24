import numpy as np

from abstractReasoning import abstractReasoningStep, validateFinalInequality
from abstractState import AbstractState, Domain, generateDomain

def generatePhaseGate(m):
    return np.array([[1, 0], [0, np.exp(2 * np.pi * 1j / (2 ** m))]], dtype=complex)

def generateControlPhaseGate(m):
    phaseGate = generatePhaseGate(m)

    zeroZero = np.array([[1, 0], [0, 0]], dtype=complex)
    oneOne = np.array([[0, 0], [0, 1]], dtype=complex)
    identity = np.identity(2)
    return np.kron(zeroZero, identity) + np.kron(oneOne, phaseGate)

def generateLinearDomain(n):
    S = generateDomain(n, Domain.LINEAR)

    initialProj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvPlusZero = np.array([[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvPlusOne = np.array([[0, 0, 0, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 0], [0, 0.5, 0, 0.5]], dtype=complex)
    initialObsvMinusZero = np.array([[0.5, 0, -0.5, 0], [0, 0, 0, 0], [-0.5, 0, 0.5, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvMinusOne = np.array([[0, 0, 0, 0], [0, 0.5, 0, -0.5], [0, 0, 0, 0], [0, -0.5, 0, 0.5]], dtype=complex)
    initialObsvZeroPlus = np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvOnePlus = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]], dtype=complex)
    initialObsvZeroMinus = np.array([[0.5, -0.5, 0, 0], [-0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvOneMinus = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, -0.5], [0, 0, -0.5, 0.5]], dtype=complex)
    initialObsvPlusPlus = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
    initialObsvMinusMinus = 0.25 * np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, -1, -1, 1]], dtype=complex)

    initialObsvs = [initialObsvPlusPlus for _ in range(n - 1)]
    initialState = AbstractState(n, S, [initialProj for _ in range(n - 1)], initialObsvs)

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)

    import pdb
    pdb.set_trace()
    from pprint import pprint

    nextState = initialState

    for i in range(n):
        nextState = abstractReasoningStep(nextState, H, [i])
        pprint(nextState.projections[0])
            
        for j in range(2, n - i):
            controlPhaseGate = generateControlPhaseGate(j)
            nextState = abstractReasoningStep(nextState, controlPhaseGate, [i, i + j - 1])
            pprint(nextState.projections[0])

    validateFinalInequality(initialState, nextState)

    print(nextState)

def generateSingleDomain(n):
    S = generateDomain(n, Domain.SINGLE)

    initialProj = np.array([[1, 0], [0, 0]], dtype=complex)
    initialObsvPlus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

    initialObsvs = [initialObsvPlus for _ in range(n)]
    initialState = AbstractState(n, S, [initialProj for _ in range(n)], initialObsvs)

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)

    # import pdb
    # pdb.set_trace()
    # from pprint import pprint

    nextState = initialState

    for i in range(n):
        nextState = abstractReasoningStep(nextState, H, [i])
            
        for j in range(2, n - i):
            controlPhaseGate = generateControlPhaseGate(j)
            nextState = abstractReasoningStep(nextState, controlPhaseGate, [i, i + j - 1])

    validateFinalInequality(initialState, nextState)

    print(nextState)
