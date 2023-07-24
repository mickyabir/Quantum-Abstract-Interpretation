import cvxpy as cp
import numpy as np

class ObjectiveFunction():
    def __init__(self):
        raise NotImplementedError

    def generateObjective(state):
        raise NotImplementedError

class TraceSum(ObjectiveFunction):
    def __init__(self):
        pass

    def generateObjective(state, fullDomain, domainIndices, U, F):
        traceSum = 0
        for i in domainIndices:
            traceSum += cp.trace(state.observables[i])

        return traceSum



