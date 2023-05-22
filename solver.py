import cvxpy as cp
import numpy as np

from constraintsUtil import getUnitRuleRHS
from abstractState import AbstractState

def generatePSDConstraint(M):
    topLeft = np.array([[1, 0], [0, 0]])
    topRight = np.array([[0, 1], [0, 0]])
    bottomLeft = np.array([[0, 0], [1, 0]])
    bottomRight = np.array([[0, 0], [0, 1]])

    finalMatrix = cp.kron(topLeft, cp.real(M))
    finalMatrix += cp.kron(topRight, -cp.imag(M))
    finalMatrix += cp.kron(bottomLeft, cp.imag(M))
    finalMatrix += cp.kron(bottomRight, cp.real(M))

    return finalMatrix


def solveUnitRuleConstraints(constraintLHS, state, fullDomain, domainIndices, gammaP, U, F):
    constraints = []

    Bs = []
    for i in domainIndices:
        domainSize = len(state.S[i])
        # state.observables[i] = cp.Variable((2 ** domainSize, 2 ** domainSize), PSD=True)
        state.observables[i] = cp.Variable((2 ** domainSize, 2 ** domainSize), hermitian=True)
        Bs.append(state.observables[i])
        I = np.identity(2 ** domainSize)

        # B_i <= I
        constraintMatrixIdentity = generatePSDConstraint(I - state.observables[i])
        constraints.append(constraintMatrixIdentity >= 0)

        # 0 <= B_i
        constraintMatrixPSD = generatePSDConstraint(state.observables[i])
        constraints.append(constraintMatrixPSD >= 0)

    constraintRHS = getUnitRuleRHS(state, U, F, gammaP, symbolic=True)
    unitRuleMatrix = constraintRHS - constraintLHS
    unitRuleMatrixRealForm = generatePSDConstraint(unitRuleMatrix)
    constraints.append(unitRuleMatrixRealForm >= 0)

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    for i in domainIndices:
        state.observables[i] = state.observables[i].value
