import numpy as np

from abstractReasoning import abstractReasoningStep, validateFinalInequality
from abstractState import AbstractState

from examples import GHZ, Miller, QFT

if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    n = 20
    GHZ.generateLinearDomain(n)
    GHZ.generateLinearDomain(n, plus=False)
    # GHZ.generateLinearDomain(n)
    # Miller.generateMiller()
    # QFT.generateSingleDomain(n)
