import numpy as np

from gates import *
from states import *

from abstractReasoning import abstractReasoningStep
from abstractState import AbstractState, Domain, generateDomain
from prover import Prover

def generate(n, domain=Domain.SINGLE):
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
