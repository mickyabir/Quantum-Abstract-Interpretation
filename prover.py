from abstractReasoning import abstractReasoningStep, validateFinalInequality
from pprint import pprint

class Prover():
    def __init__(self, initialState):
        self.initialState = initialState
        self.currentState = initialState
        self.states = [initialState]
        self.operations = []
        self.opPos = 0

    def addOp(self, U, F):
        self.operations.append([U, F])

    def apply(self, proofRule=None):
        if self.opPos >= len(self.operations):
            return False

        U, F = self.operations[self.opPos]
        self.currentState = abstractReasoningStep(self.currentState, U, F)
        self.states.append(self.currentState)
        self.opPos += 1

        return True

    def backtrack(self):
        # Can't backtrack past initial state
        if len(self.states) == 1:
            return

        self.states.pop()
        self.currentState = self.states[-1]
        self.opPos -= 1

    def print(self):
        print(self.currentState)

    def proj(self, pos):
        if type(pos) == int:
            print(self.currentState.S[pos])
            pprint(self.currentState.projections[pos])
        elif type(pos) == list:
            # pos is a list of qubits of interest
            for i in range(len(self.currentState.S)):
                s = self.currentState.S[i]
                if not set(s).isdisjoint(pos):
                    print(s)
                    pprint(self.currentState.projections[i])
                    print()
                    
    def obsv(self, pos):
        if type(pos) == int:
            print(self.currentState.S[pos])
            pprint(self.currentState.observables[pos])
        elif type(pos) == list:
            # pos is a list of qubits of interest
            for i in range(len(self.currentState.S)):
                s = self.currentState.S[i]
                if not set(s).isdisjoint(pos):
                    print(s)
                    pprint(self.currentState.observables[i])
                    print()

    def validate(self):
        validateFinalInequality(self.initialState, self.currentState)
