import numpy as np

from gates import *
from states import *

from abstractReasoning import abstractReasoningStep
from abstractState import AbstractState, Domain, generateDomain
from prover import Prover

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

    initialProj = generateDensityMatrixFromQubits([Zero])
    initialObsv = generateDensityMatrixFromQubits([Plus])

    initialObsvs = [initialObsv for _ in range(n)]
    initialProjs = [initialProj for _ in range(n)]
    initialState = AbstractState(n, S, initialProjs, initialObsvs)

    prover = Prover(initialState)

    for i in range(n):
        prover.addOp(H, [i])
            
        for j in range(2, n - i):
            controlPhaseGate = generateControlPhaseGate(j)
            prover.addOp(controlPhaseGate, [i, i + j - 1])

    while prover.apply():
        continue

    prover.validate()
    prover.print()
