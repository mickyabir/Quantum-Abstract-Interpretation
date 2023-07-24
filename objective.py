import cvxpy as cp
import numpy as np

class ObjectiveFunction():
    def __init__(self):
        raise NotImplementedError

    def generateObjective(state):
        raise NotImplementedError

class ZeroObjective(ObjectiveFunction):
    def __init__(self):
        pass

    def generateObjective(self, state, fullDomain, domainIndices, gammaP, U, F):
        return 0

class TraceSum(ObjectiveFunction):
    def __init__(self):
        pass

    def generateObjective(self, state, fullDomain, domainIndices, gammaP, U, F):
        traceSum = 0
        for i in domainIndices:
            traceSum += cp.trace(state.observables[i])

        return traceSum

