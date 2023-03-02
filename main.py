import z3

import numpy as np

from enum import Enum

class Rules(Enum):
    SKIP-AX = 1
    UNIT = 2
    SEQ = 3
    CON = 4

class Prover():
    def __init__(self):
        pass

    def prove_formula(self, formula):
        s = z3.Solver()
        s.add(z3.Not(formula))
        res = s.check()
        if str(res) == "unsat":
            print("Valid", formula)
            return True
        else:
            print("Not valid", formula)
            return False

    def prove_triple(self, precond, command, postcond):
        if command.type == Rules.SKIP-AX:
            formula = z3.Implies(precond, postcond)
            print("SKIP: Trying to prove", formula)
            res = self.prove_formula(formula)
            return res

        elif command.type == Rules.UNIT:
            pass

    def prove_PSD(self, matrix):
        return np.all(np.linalg.eigvals(matrix) > 0)

print(Rules.UNIT)

x = np.matrix([[1, 0], [0, 1]])

prover = Prover()

print(prover.prove_PSD(x))
print(prover.prove_formula(True))
