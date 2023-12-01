import qassist

import numpy as np

from gates import *
from states import *

from abstractState import AbstractState, Domain, generateDomain
from util import computeSubspaceProjection

def computeInequality(obsA, obsB):
    zzVec = generateTensorState([Zero, Zero])
    zzVecLeft = zzVec.reshape((1, 4))
    zzVecRight = zzVec.reshape((4, 1))

    ooVec = generateTensorState([One, One])
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

def generate(n, config):
    domain = config.get('domain')
    plus = config.get('plus')
    noisy = config.get('noisy')

    if domain is None:
        domain = Domain.LINEAR

    if plus is None:
        plus = True

    S = generateDomain(n, domain)
    if domain == Domain.LINEAR:
        initialProj = generateDensityMatrixFromQubits([Zero, Zero])
        initialProjs = [initialProj for _ in range(n)]

        if plus:
        # |a|^2 >= 0.5
            initialObsvPlusZero = generateDensityMatrixFromQubits([Plus, Zero])
            initialObsvPlusPlus = generateDensityMatrixFromQubits([Plus, Plus])
            initialObsvs = [initialObsvPlusZero] + [initialObsvPlusPlus for _ in range(n - 1)]
        else:
            # |a|^2 <= 0.5
            initialObsvMinusZero = generateDensityMatrixFromQubits([Minus, Zero])
            initialObsvMinusMinus = generateDensityMatrixFromQubits([Minus, Minus])
            initialObsvs = [initialObsvMinusZero] + [initialObsvMinusMinus for _ in range(n - 1)]
    else:
        raise NotImplementedError

    eps = 0.000001

    if noisy is not None and noisy:
        H_gate = generateNaiveNoisyGate(H, eps)
        CNOT_gate = generateNaiveNoisyGate(CNOT, eps)
        T_gate = generateNaiveNoisyGate(T, eps)
    else:
        H_gate = H
        CNOT_gate = CNOT
        T_gate = T

    ops = []
    ops.append([H_gate, [0]])

    for i in range(1, n):
        ops.append([CNOT_gate, [0, i]])

    for i in range(n):
            ops.append([T_gate, [i]])

    return qassist.Program(n, S, initialProjs, initialObsvs, ops)

def proof(prover):
    while prover.apply():
        prover.print()
        continue

    prover.validate()

    initialObsvs = prover.initialState.observables
    computeInequality(initialObsvs, prover.currentState.observables)

    psi, bound = computeSubspaceProjection(prover.initialState, prover.currentState)
    print(psi)
    print(bound)
