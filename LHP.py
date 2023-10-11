import qassist

import numpy as np

from abstractReasoning import abstractReasoningStep, validateFinalInequality
from abstractState import AbstractState, Domain
from prover import Prover

from examples import GHZ, Miller, QFT

if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    n = 3

    qassist.prove(n, GHZ.generate, GHZ.proof, None, config={'plus':True})
    # qassist.interactive(n, GHZ.generate, None, config={'plus':False})

    # qassist.prove(n, Miller.generate, Miller.proof, None, config={})
    # qassist.interactive(n, Miller.generate, None, config={})

    # qassist.prove(n, QFT.generate, QFT.proof, None, config={})
    # qassist.interactive(n, QFT.generate, None, config={})
