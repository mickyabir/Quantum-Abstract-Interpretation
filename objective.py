import cvxpy as cp
import numpy as np

class ObjectiveFunction():
    def generateObjective(state):
        raise NotImplementedError

    def name():
        raise NotImplementedError

class ZeroObjective(ObjectiveFunction):
    def generateObjective(self, state, fullDomain, domainIndices, gammaP, U, F):
        return 0

    def name():
        return 'zero'

class TraceSum(ObjectiveFunction):
    def generateObjective(self, state, fullDomain, domainIndices, gammaP, U, F):
        traceSum = 0
        for i in domainIndices:
            traceSum += cp.trace(state.observables[i])

        return traceSum

    def name():
        return 'tracesum'

class LogDet(ObjectiveFunction):
    def generateObjective(self, state, fullDomain, domainIndices, gammaP, U, F):
        sumLogDet = 0
        for i in domainIndices:
            sumLogDet += cp.log_det(state.observables[i])

        return sumLogDet

    def name():
        return 'logdet'

class LambdaMax(ObjectiveFunction):
    def generateObjective(self, state, fullDomain, domainIndices, gammaP, U, F):
        sumLambdaMax = 0
        for i in domainIndices:
            sumLambdaMax += cp.lambda_max(state.observables[i])

        return sumLambdaMax

    def name():
        return 'lambdamax'

class MaxDiagonalElem(ObjectiveFunction):
    def generateObjective(self, state, fullDomain, domainIndices, gammaP, U, F):
        maxDiagonalSum = 0
        for i in domainIndices:
            maxElem = 0
            for j in range(len(state.observables[i].shape[0])):
                maxDiagonalSum += cp.max(maxElem, state.observables[i][j][j])

        return maxDiagonalSum

    def name():
        return 'maxdiag'

objectiveFunctionMap = {
        ZeroObjective.name(): ZeroObjective(),
        TraceSum.name(): TraceSum(),
        LogDet.name(): LogDet(),
        LambdaMax.name(): LambdaMax(),
        MaxDiagonalElem.name(): MaxDiagonalElem(),
}
