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
def generate(n, config={'U': T, 'psiGates': [[X, [0]]], 't': 3}):
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

    qftOps = []
    for i in range(t):
        qftOps.append([H, [t]])
            
        for j in range(2, t - i + 1):
            controlPhaseGate = generateControlPhaseGate(j, inverse=True)
            qftOps.append([controlPhaseGate, [i, i + j - 1]])

    qftOps = qftOps[::-1]

    ops += qftOps

    return qassist.Program(n + t, S, initialProjs, initialObsvs, ops)

def proof(prover):
    while prover.apply():
        continue

    prover.validate()
    prover.print()

    psi, bound = computeSubspaceProjection(prover.initialState, prover.currentState)
    print(psi)
    print(bound)
