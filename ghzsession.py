import qassist

from gates import *
from states import *

def generator(n, config={}):
    S = [[i] for i in range(n)]

    initialProj = generateDensityMatrixFromQubits([Zero])
    initialObsv = generateDensityMatrixFromQubits([Plus])
    initialProjs = [initialProj for _ in range(n)]
    initialObsvs = [initialObsv for _ in range(n)]

    ops = []
    ops.append([H, [0]])

    for i in range(1, n):
        ops.append([CNOT, [0, i]])

    for i in range(n):
            ops.append([T, [i]])

    return qassist.Program(n, S, initialProjs, initialObsvs, ops)

def proof(prover):
    while prover.apply():
        continue

    prover.validate()
    prover.print()

if __name__ == '__main__':
    n = 3
    config = {}
    # qassist.prove(n, generator, proof, config)
    qassist.interactive(n, generator, config)
