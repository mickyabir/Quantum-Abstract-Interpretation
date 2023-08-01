import numpy as np

from gates import *
from states import *

from abstractState import AbstractState

def generateMiller():
    n = 3

    # S = []
    # for i in range(n - 1):
    #   S.append([i, i + 1])

    # initialProj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    # initialObsv = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
    # initialState = AbstractState(n, S, [initialProj for _ in range(n - 1)], [initialObsv for _ in range(n-1)])

    # FULL ABSTRACT DOMAIN
    S = [[0, 1, 2]]
    initialProj = generateDensityMatrixFromQubits([Zero, Zero, Zero])
    initialObsv = generateDensityMatrixFromQubits([Plus, Plus, Plus])
    initialState = AbstractState(n, S, [initialProj], [initialObsv])

    ops = []
    ops.append([CNOT10, [1, 2]])
    ops.append([H, [2]])
    ops.append([T, [1]])
    ops.append([T, [0]])
    ops.append([T, [2]])
    ops.append([CNOT, [0, 1]])
    ops.append([CNOT10, [0, 2]])
    ops.append([CNOT, [1, 2]])
    ops.append([TDG, [0]])
    ops.append([CNOT10, [1, 0]])
    ops.append([TDG, [1]])
    ops.append([TDG, [0]])
    ops.append([T, [2]])
    ops.append([CNOT10, [0, 2]])
    ops.append([CNOT, [1, 2]])
    ops.append([CNOT, [0, 1]])
    ops.append([H, [2]])
    ops.append([H, [0]])
    ops.append([T, [2]])
    ops.append([T, [1]])
    ops.append([T, [0]])
    ops.append([CNOT, [1, 2]])
    ops.append([CNOT, [0, 1]])
    ops.append([CNOT10, [0, 2]])
    ops.append([TDG, [1]])
    ops.append([CNOT10, [1, 2]])
    ops.append([TDG, [2]])
    ops.append([TDG, [1]])
    ops.append([T, [0]])
    ops.append([CNOT, [0, 1]])
    ops.append([CNOT10, [0, 2]])
    ops.append([CNOT, [1, 2]])
    ops.append([H, [0]])
    ops.append([H, [2]])
    ops.append([T, [1]])
    ops.append([T, [0]])
    ops.append([T, [2]])
    ops.append([CNOT, [0, 1]])
    ops.append([CNOT10, [0, 2]])
    ops.append([CNOT, [1, 2]])
    ops.append([TDG, [0]])
    ops.append([CNOT10, [1, 0]])
    ops.append([TDG, [1]])
    ops.append([TDG, [0]])
    ops.append([T, [2]])
    ops.append([CNOT10, [0, 2]])
    ops.append([CNOT, [1, 2]])
    ops.append([CNOT, [0, 1]])
    ops.append([H, [2]])
    ops.append([CNOT10, [1, 2]])

    return initialState, ops

def proof(prover):
    while prover.apply():
        continue

    prover.validate()
    prover.print()
