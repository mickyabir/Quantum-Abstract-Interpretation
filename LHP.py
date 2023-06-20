import numpy as np

from abstractReasoning import abstractReasoningStep
from abstractState import AbstractState

def validateFinalInequality(initialState, finalState):
    sumInitial = 0
    sumFinal = 0

    for i in range(len(initialState.S)):
        sumInitial += np.trace(initialState.projections[i] @ initialState.observables[i])
        sumFinal += np.trace(finalState.projections[i] @ finalState.observables[i])

    assert(sumInitial <= sumFinal)

def computeInequalityGHZ(obsA, obsB):
    zVec = np.array([1, 0], dtype=complex)
    zzVec = np.kron(zVec, zVec)
    zzVecLeft = zzVec.reshape((1, 4))
    zzVecRight = zzVec.reshape((4, 1))

    oVec = np.array([0, 1], dtype=complex)
    ooVec = np.kron(oVec, oVec)
    ooVecLeft = ooVec.reshape((1, 4))
    ooVecRight = ooVec.reshape((4, 1))

    sumA = 0
    sumB00 = 0
    sumB11 = 0
    for i in range(len(obsA)):
        sumA += zzVecLeft @ obsA[i] @ zzVecRight
        sumB00 += zzVecLeft @ obsB[i] @ zzVecRight
        sumB11 += ooVecLeft @ obsB[i] @ ooVecRight

    bound00 = np.round(np.real((sumA - sumB11) / (sumB00 - sumB11))[0][0], 5)
    b00eq = '<=' if (sumB00 - sumB11) < 0 else '>='
    bound11 = np.round(np.real((sumA - sumB00) / (sumB11 - sumB00))[0][0], 5)
    b11eq = '<=' if (sumB11 - sumB00) < 0 else '>='

    print(f'|a|^2 {b00eq} {bound00}')
    print(f'|b|^2 {b11eq} {bound11}')


def generateGHZPaperPartial(n):
    S = []

    for i in range(n - 1):
        S.append([i, i + 1])

    initialProj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvPlusZero = np.array([[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvPlus = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
    initialObsvMinus = 0.25 * np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, -1, -1, 1]], dtype=complex)
    # initialObsvs = [initialObsvPlus for _ in range(n-1)]
    # initialObsvs = [initialObsvPlusZero, initialObsvPlus]
    initialObsvs = [initialObsvPlusZero] + [initialObsvPlus for _ in range(n - 2)]

    initialState = AbstractState(n, S, [initialProj for _ in range(n - 1)], initialObsvs)

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = abstractReasoningStep(initialState, H, [0])

    for i in range(1, n):
        try:
            nextState = abstractReasoningStep(nextState, CNOT, [0, i])
            # print(nextState)
        except:
            import pdb
            pdb.set_trace()
            nextState = abstractReasoningStep(nextState, CNOT, [0, i])
            print(nextState)

    print(nextState)

    validateFinalInequality(initialState, nextState)

    computeInequalityGHZ(initialObsvs, nextState.observables)

    # import pdb
    # pdb.set_trace()

def generateGHZPaperFull(n, plus=True):
    S = []

    for i in range(n):
        for j in range(i + 1, n):
            S.append([i, j])

    initialProj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvPlusZero = np.array([[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsvPlus = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
    initialObsvMinus = 0.25 * np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, -1, -1, 1]], dtype=complex)
    zeroMat = initialObsvPlus

    if plus:
        initialObsvs = [initialObsvPlus for _ in range(int(n * (n - 1) / 2))]
    else:
        initialObsvs = [initialObsvMinus for _ in range(int(n * (n - 1) / 2))]

    # initialState = AbstractState(n, S, [initialProj for _ in range(int(n * (n - 1) / 2))], initialObsvs)
    initialState = AbstractState(n, S, [initialProj for _ in range(int(n * (n - 1) / 2))], [initialObsvPlusZero, initialObsvPlusZero, zeroMat])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    import pdb
    pdb.set_trace()
    nextState = abstractReasoningStep(initialState, H, [0])

    for i in range(1, n):
        try:
            nextState = abstractReasoningStep(nextState, CNOT, [0, i])
            # print(nextState)
        except:
            import pdb
            pdb.set_trace()
            nextState = abstractReasoningStep(nextState, CNOT, [0, i])
            print(nextState)

    # print(nextState)

    validateFinalInequality(initialState, nextState)

    computeInequalityGHZ(initialObsvs, nextState.observables)

def generateMiller():
    n = 3
    S = []

    for i in range(n - 1):
        S.append([i, i + 1])

    initialProj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initialObsv = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
    initialState = AbstractState(n, S, [initialProj for _ in range(n - 1)], [initialObsv for _ in range(n-1)])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    T = np.array([[1, 0],[0, np.exp(1j * np.pi / 4)]], dtype=complex)
    TDG = np.array([[1, 0],[0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CNOT10 = np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)

    nextState = initialState

    # import pdb
    # pdb.set_trace()

    nextState = abstractReasoningStep(initialState, CNOT10, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [1, 0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)

    # import pdb
    # pdb.set_trace()
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [1, 0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)

    import pdb
    pdb.set_trace()
    nextState = abstractReasoningStep(nextState, H, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [1, 2])
    print(nextState)

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    # generateMiller()

    n = 3
    generateGHZPaperPartial(n)
    # generateGHZPaperFull(n, plus=True)
