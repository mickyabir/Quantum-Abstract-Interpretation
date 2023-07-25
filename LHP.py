import numpy as np

from abstractReasoning import abstractReasoningStep, validateFinalInequality
from abstractState import AbstractState, Domain

from examples import GHZ, Miller, QFT

if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    n = 3
    GHZ.generate(n)
    GHZ.generate(n, plus=False)
    # GHZ.generate(n)
    # Miller.generateMiller()
    # QFT.generate(n, domain=Domain.SINGLE)
    # QFT.generate(n, domain=Domain.LINEAR)
