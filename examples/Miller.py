import numpy as np
import qassist

from gates import *
from states import *

from abstractState import AbstractState

def generateMiller(n, config={}):
    n = 3

    S = []
    # #for i in range(n - 1):
    # #  S.append([i, i + 1])
    # S = [[0,1], [1,2], [0,2]]

    # initialProj = generateDensityMatrixFromQubits([Zero, Zero])
    # initialObsv = generateDensityMatrixFromQubits([Plus, Plus])
    # # initialProjs = [initialProj for _ in range(n - 1)]
    # # initialObsvs = [initialObsv for _ in range(n - 1)]
    # initialProjs = [initialProj for _ in range(3)]
    # initialObsvs = [initialObsv for _ in range(3)]

    # FULL ABSTRACT DOMAIN
    S = [[0, 1, 2]]
    initialProj = generateDensityMatrixFromQubits([Zero, Zero, Zero])
    initialObsv = generateDensityMatrixFromQubits([Plus, Plus, Plus])
    initialProjs = [initialProj]
    initialObsvs = [initialObsv]

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

    return qassist.Program(n, S, initialProjs, initialObsvs, ops)

# Returns None if lemma doesn't apply
# Otherwise return new observables
def lemma1(n, observables):
    import pdb
    pdb.set_trace()
    return None

def proof(prover):
    lemmas = {'lemma1': lemma1}
    n = 3
    generator = generateMiller
    config = {}
    qassist.interactive(n, generator, lemmas, config)
    return

    while prover.apply():
        continue

    prover.validate()
    prover.print()


