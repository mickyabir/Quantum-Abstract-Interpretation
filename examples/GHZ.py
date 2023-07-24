import numpy as np

from gates import *
from states import *

from abstractReasoning import abstractReasoningStep, validateFinalInequality
from abstractState import AbstractState, Domain, generateDomain
from prover import Prover

def computeInequality(obsA, obsB):
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

def generateLinearDomain(n, plus=True):
    S = generateDomain(n, Domain.LINEAR)

    initialProj = generateDensityMatrixFromQubits([Zero, Zero])
    initialProjs = [initialProj for _ in range(n - 1)]

    if plus:
    # |a|^2 >= 0.5
        initialObsvPlusZero = generateDensityMatrixFromQubits([Plus, Zero])
        initialObsvPlusPlus = generateDensityMatrixFromQubits([Plus, Plus])
        initialObsvs = [initialObsvPlusZero] + [initialObsvPlusPlus for _ in range(n - 2)]
    else:
        # |a|^2 <= 0.5
        initialObsvMinusZero = generateDensityMatrixFromQubits([Minus, Zero])
        initialObsvMinusMinus = generateDensityMatrixFromQubits([Minus, Minus])
        initialObsvs = [initialObsvMinusZero] + [initialObsvMinusMinus for _ in range(n - 2)]

    initialState = AbstractState(n, S, initialProjs, initialObsvs)

    prover = Prover(initialState)
    prover.addOp(H, [0])

    for i in range(1, n):
        prover.addOp(CNOT, [0, i])

    import random
    for i in range(n):
            prover.addOp(T, [i])

    while prover.apply():
        continue

    prover.validate()
    computeInequality(initialObsvs, prover.currentState.observables)

