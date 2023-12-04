import qassist

import numpy as np

from gates import *
from states import *

from abstractState import AbstractState, Domain, generateDomain

def generate(n, config):
    domain = config.get('domain')
    inverse = config.get('inverse')

    if not domain:
        domain = Domain.SINGLE
    if not inverse:
        inverse = False

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

    ops = []

    for i in range(n):
        ops.append([H, [i]])
            
        for j in range(2, n - i + 1):
            controlPhaseGate = generateControlPhaseGate(j, inverse)
            ops.append([controlPhaseGate, [i, i + j - 1]])


    for i in range(int(np.floor(n / 2))):
        ops.append([SWAP, [i, n - i - 1]])

    if inverse:
        ops = ops[::-1]

    return qassist.Program(n, S, initialProjs, initialObsvs, ops)

def proof(prover):
    while prover.apply():
        continue

    prover.validate()
    prover.print()
