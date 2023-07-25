import numpy as np

from abstractReasoning import abstractReasoningStep, validateFinalInequality
from abstractState import AbstractState, Domain
from prover import Prover

from examples import GHZ, Miller, QFT

if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    n = 3

    # initialState, ops = GHZ.generate(n)
    # prover = Prover(initialState, ops)
    # GHZ.proof(prover)
    # 
    # initialState, ops = GHZ.generate(n, plus=False)
    # prover = Prover(initialState, ops)
    # GHZ.proof(prover)

    initialState, ops = Miller.generateMiller()
    prover = Prover(initialState, ops)
    Miller.proof(prover)

    # initialState, ops = QFT.generate(n, domain=Domain.SINGLE)
    # prover = Prover(initialState, ops)
    # QFT.proof(prover)

    # initialState, ops = QFT.generate(n, domain=Domain.LINEAR)
    # prover = Prover(initialState, ops)
    # QFT.proof(prover)
