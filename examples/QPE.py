import qassist

import numpy as np

from gates import *
from states import *

from abstractState import AbstractState
from util import computeSubspaceProjection

# n is the dimension of the unitary U
# psiGates is the list of gates to apply to |00...0> to prepare |psi>
# t is the number of counting qubits
# NOTE: assumes n = 1 for now
def generateFront(n, config={'U': T, 'psiGates': [[X, [0]]], 't': 3}):
    U = config['U']
    psiGates = config['psiGates']
    t = config['t']

    S = []
    for i in range(n + t - 1):
      S.append([i, i + 1])

    initialProj = generateDensityMatrixFromQubits([Zero, Zero])
    initialObsv = generateDensityMatrixFromQubits([Plus, Plus])
    initialProjs = [initialProj for _ in range(n + t - 1)]
    initialObsvs = [initialObsv for _ in range(n + t - 1)]

    ops = []

    for psiGate in psiGates:
        M, qubits = psiGate
        qubits = [x + t for x in qubits]
        ops.append([M, qubits])
    
    for i in range(t):
        ops.append([H, [i]])

    for i in range(t):
        cU = generateControlUGate(U)
        for j in range(2 ** (t - 1 - i)):
            ops.append([cU, [i, t]])

    return qassist.Program(n + t, S, initialProjs, initialObsvs, ops)

# n is the dimension of the unitary U
# psiGates is the list of gates to apply to |00...0> to prepare |psi>
# t is the number of counting qubits
# NOTE: assumes n = 1 for now
def generateBack(n, config={'U': T, 'psiGates': [[X, [0]]], 't': 3}):
    U = config['U']
    psiGates = config['psiGates']
    t = config['t']

    S = []
    for i in range(n + t - 1):
      S.append([i, i + 1])

    initialProj = generateDensityMatrixFromQubits([Zero, Zero])
    initialObsv = generateDensityMatrixFromQubits([Plus, Plus])
    initialProjs = [initialProj for _ in range(n + t - 1)]
    initialObsvs = [initialObsv for _ in range(n + t - 1)]

    ops = []

    ops = []
    for i in range(t):
        ops.append([H, [t]])
            
        for j in range(2, t - i + 1):
            controlPhaseGate = generateControlPhaseGate(j, inverse=True)
            ops.append([controlPhaseGate, [i, i + j - 1]])

    for i in range(int(np.floor(t / 2))):
        ops.append([SWAP, [i, t - i - 1]])

    return qassist.Program(n + t, S, initialProjs, initialObsvs, ops)

def proof(proverFront, proverBack):
    while proverFront.apply():
        continue

    while proverBack.apply():
        continue

    proverFront.validate()
    proverBack.validate()
    proverFront.print()
    proverBack.print()

    # psi, bound = computeSubspaceProjection(prover.initialState, prover.currentState)
    # print(psi)
    # print(bound)
