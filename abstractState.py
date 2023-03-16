from pprint import pprint, pprint_repr

class AbstractState():
    def __init__(self, n, S, projections):
        self.n = n
        self.S = S
        self.projections = projections

    def __repr__(self):
        retStr = ""
        retStr += f"Abstract State:\n\nn: {self.n}\n"
        for i in range(len(self.projections)):
            retStr += f"\n{self.S[i]}\n<->\n{pprint_repr(self.projections[i])}\n\n"
        return retStr


