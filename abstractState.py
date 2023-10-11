from enum import Enum

from pprint import pprint, pprint_repr

class Domain(Enum):
    SINGLE = 1
    LINEAR = 2

def generateDomain(n, domain=Domain.LINEAR):
    if domain == Domain.SINGLE:
        return [[i] for i in range(n)]
    elif domain == Domain.LINEAR:
        S = []

        for i in range(n - 1):
            S.append([i, i + 1])

        return S + [[0, n - 1]]
    else:
        raise NotImplementedError

class AbstractState():
    def __init__(self, n, S, projections, observables = None):
        self.n = n
        self.S = S
        self.projections = projections
        self.observables = observables

    def __repr__(self):
        retStr = ""
        retStr += f"Abstract State:\n\nn: {self.n}\n\n"

        retStr += f"Projections:\n"
        for i in range(len(self.projections)):
            retStr += f"\n{self.S[i]}\n<->\n{pprint_repr(self.projections[i])}\n\n"

        if self.observables is not None:
            retStr += f"Observables:\n"
            for i in range(len(self.projections)):
                retStr += f"\n{self.S[i]}\n<->\n{pprint_repr(self.observables[i])}\n\n"
        return retStr


