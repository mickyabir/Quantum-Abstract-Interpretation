import cvxpy as cp
import numpy as np

from abstractState import AbstractState
from constraintsUtil import getUnitRuleRHS
from matrixUtil import truncateComplexObject
from objective import TraceSum

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

def solveUnitRuleConstraints(constraintLHS, state, fullDomain, domainIndices, gammaP, U, F, objectiveFunction=None):
    constraints = []

    Bs = []
    for i in domainIndices:
        domainSize = len(state.S[i])
        state.observables[i] = cp.Variable((2 ** domainSize, 2 ** domainSize), hermitian=True)
        Bs.append(state.observables[i])
        I = np.identity(2 ** domainSize)

        # B_i <= I
        M = I - state.observables[i]
        constraintMatrixIdentity = generatePSDConstraint(M)
        constraints.append(constraintMatrixIdentity >= 0)

        # B_i <= Q_i
        M = state.projections[i] - state.observables[i]
        constraintMatrixIdentity = generatePSDConstraint(M)
        constraints.append(constraintMatrixIdentity >= 0)

        # 0 <= B_i
        M = state.observables[i]
        constraintMatrixPSD = generatePSDConstraint(M)
        constraints.append(constraintMatrixPSD >= 0)

    constraintRHS = getUnitRuleRHS(state, U, F, gammaP, backend=cp)
    unitRuleMatrix = constraintRHS - constraintLHS
    unitRuleMatrixRealForm = generatePSDConstraint(unitRuleMatrix)
    constraints.append(unitRuleMatrixRealForm >= 0)

    if objectiveFunction is None:
        objectiveFunction = TraceSum()
    objective = objectiveFunction.generateObjective(state, fullDomain, domainIndices, gammaP, U, F)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    # prob.solve(solver='CVXOPT')
    # prob.solve(solver='CBC')
    prob.solve(solver='ECOS_BB')

    for i in domainIndices:
        state.observables[i] = truncateComplexObject(state.observables[i].value)
