import numpy as np

from gates import *
from states import *

from abstractState import AbstractState, Domain, generateDomain

def generate(n, domain=Domain.SINGLE, inverse=False):
    S = generateDomain(n, domain)

    if domain == Domain.SINGLE:
        initialProj = generateDensityMatrixFromQubits([Zero])
        initialObsv = generateDensityMatrixFromQubits([Plus])
        initialObsvs = [initialObsv for _ in range(n)]
        initialProjs = [initialProj for _ in range(n)]
    elif domain == Domain.LINEAR:
        initialProj = generateDensityMatrixFromQubits([Zero, Zero])
        initialObsv = generateDensityMatrixFromQubits([Plus, Plus])
        initialProjs = [initialObsv for _ in range(n - 1)]
        initialObsvs = [initialObsv for _ in range(n - 1)]
    else:
        raise NotImplementedError

    initialState = AbstractState(n, S, initialProjs, initialObsvs)

    ops = []

    for i in range(n):
        ops.append([H, [i]])
            
        for j in range(2, n - i + 1):
            controlPhaseGate = generateControlPhaseGate(j, inverse)
            ops.append([controlPhaseGate, [i, i + j - 1]])

    if inverse:
        ops = ops[::-1]

    return initialState, ops

def proof(prover):
    while prover.apply():
        continue

    prover.validate()
    prover.print()
