from abstractReasoning import abstractReasoningStep, validateFinalInequality
from constraintsUtil import getFullDomain
from pprint import pprint

class Prover():
    def __init__(self, initialState, ops, lemmas):
        self.initialState = initialState
        self.currentState = initialState
        self.states = [initialState]
        self.operations = ops
        self.opPos = 0
        self.lemmas = lemmas

    def getCurrentOp(self):
        if self.opPos < len(self.operations):
            U, F = self.operations[self.opPos]
        else:
            U, F = None, None
        return U, F

    def getPrevOp(self):
        if self.opPos <= 0:
            return self.operations[0]
        return self.operations[self.opPos - 1]

    #TODO: allow adding contraints
    def apply(self, objectiveFunction=None, constraints=None):
        if self.opPos >= len(self.operations):
            return False

        U, F = self.operations[self.opPos]
        self.currentState = abstractReasoningStep(self.currentState, U, F, objectiveFunction)
        self.states.append(self.currentState)
        self.opPos += 1

        return True

    def lemma(self, lemmaName):
        U, F = self.operations[self.opPos]
        _, domainIndices = getFullDomain(self.currentState, F)
        n = self.currentState.n
        workingSet = [self.currentState.observables[i] for i in domainIndices]
        lemmaApplied = self.lemmas[lemmaName](n, workingSet)
        if lemmaApplied:
            self.currentState = abstractReasoningStep(self.currentState, U, F, objectiveFunction)
            self.states.append(self.currentState)
            self.opPos += 1

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
        print('Projections:\n')
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
        print('Observables:\n')
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
