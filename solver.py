import cvxpy as cp
import numpy as np

from abstractState import AbstractState
from constraintsUtil import getUnitRuleRHS
from matrixUtil import truncateComplexObject

def generatePSDConstraint(M, symbolic):
    topLeft = np.array([[1, 0], [0, 0]])
    topRight = np.array([[0, 1], [0, 0]])
    bottomLeft = np.array([[0, 0], [1, 0]])
    bottomRight = np.array([[0, 0], [0, 1]])

    finalMatrix = None

    if symbolic:
        finalMatrix = cp.kron(topLeft, cp.real(M))
        finalMatrix += cp.kron(topRight, -cp.imag(M))
        finalMatrix += cp.kron(bottomLeft, cp.imag(M))
        finalMatrix += cp.kron(bottomRight, cp.real(M))
    else:
        finalMatrix = np.kron(topLeft, np.real(M))
        finalMatrix += np.kron(topRight, -np.imag(M))
        finalMatrix += np.kron(bottomLeft, np.imag(M))
        finalMatrix += np.kron(bottomRight, np.real(M))

    return finalMatrix

def solveUnitRuleConstraints(constraintLHS, state, fullDomain, domainIndices, gammaP, U, F):
    constraints = []

    Bs = []
    for i in domainIndices:
        domainSize = len(state.S[i])
        state.observables[i] = cp.Variable((2 ** domainSize, 2 ** domainSize), hermitian=True)
        Bs.append(state.observables[i])
        I = np.identity(2 ** domainSize)

        # B_i <= I
        constraintMatrixIdentity = generatePSDConstraint(I - state.observables[i], symbolic=True)
        constraints.append(constraintMatrixIdentity >= 0)

        # 0 <= B_i
        constraintMatrixPSD = generatePSDConstraint(state.observables[i], symbolic=True)
        constraints.append(constraintMatrixPSD >= 0)

    constraintRHS = getUnitRuleRHS(state, U, F, gammaP, symbolic=True)
    unitRuleMatrix = constraintRHS - constraintLHS
    unitRuleMatrixRealForm = generatePSDConstraint(unitRuleMatrix, symbolic=True)
    constraints.append(unitRuleMatrixRealForm >= 0)

    prob = cp.Problem(cp.Minimize(0), constraints)
    # prob.solve(solver='ECOS_BB')
    # prob.solve(solver='CVXOPT')
    prob.solve(solver='CBC')

    for i in domainIndices:
        state.observables[i] = truncateComplexObject(state.observables[i].value)
