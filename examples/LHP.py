import numpy as np

from abstractReasoning import abstractReasoningStep, validateFinalInequality
from abstractState import AbstractState

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
    initialObsvs = [initialObsvPlusZero] + [initialObsvPlus for _ in range(n - 2)]

    initialState = AbstractState(n, S, [initialProj for _ in range(n - 1)], initialObsvs)

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = abstractReasoningStep(initialState, H, [0])

    for i in range(1, n):
        try:
            nextState = abstractReasoningStep(nextState, CNOT, [0, i])
        except:
            import pdb
            pdb.set_trace()
            nextState = abstractReasoningStep(nextState, CNOT, [0, i])
            print(nextState)

    print(nextState)

    validateFinalInequality(initialState, nextState)

    computeInequalityGHZ(initialObsvs, nextState.observables)

if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    n = 300
    generateGHZPaperPartial(n)
