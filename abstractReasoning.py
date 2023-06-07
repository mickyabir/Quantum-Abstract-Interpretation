import numpy as np

from abstractStep import abstractStep
from constraintsUtil import getFullDomain, getUnitRuleLHS, getUnitRuleRHS, fullDomainProjectionExpansion, fullDomainObservableExpansion
from matrixUtil import isSemidefinitePositive, zero_tol, truncateComplexObject
from solver import solveUnitRuleConstraints

def verifyUnitRule(stateP, stateQ, U, F):
    constraintLHS = getUnitRuleLHS(stateP, F)

    fullDomain, domainIndices = getFullDomain(stateP, F)
    forwardMap = {fullDomain[i]:i for i in range(len(fullDomain))}
    gammaP = fullDomainProjectionExpansion(fullDomain, stateP, forwardMap)
    constraintRHS = getUnitRuleRHS(stateQ, U, F, gammaP)

    constraint = truncateComplexObject(constraintRHS - constraintLHS)
    constraint.real[abs(constraint.real) < zero_tol] = 0.0
    constraint.imag[abs(constraint.imag) < zero_tol] = 0.0
    verifySDP = isSemidefinitePositive(constraint)

    if not verifySDP:
        eigvals, eigvecs = np.linalg.eig(constraint)
        lamb = np.diag(eigvals)
        lamb[lamb < 0] = 0 + 0j
        newConstraint = eigvecs @ lamb @ np.linalg.inv(eigvecs)

        assert(isSemidefinitePositive(newConstraint))

        newObservable = newConstraint + constraintLHS

        assert(len(domainIndices) == 1, 'Can only fix if M_B = B_i for now')

        stateQ.observables[domainIndices[0]] = newObservable

        # Verify new observable
        constraintRHS = getUnitRuleRHS(stateQ, U, F, gammaP)
        constraint = truncateComplexObject(constraintRHS - constraintLHS)
        constraint.real[abs(constraint.real) < zero_tol] = 0.0
        constraint.imag[abs(constraint.imag) < zero_tol] = 0.0
        verifySDP = isSemidefinitePositive(constraint)

        # Last ditch effort
        if not verifySDP:
            stateQ.observables[domainIndices[0]] = stateP.observables[domainIndices[0]]

            # Verify new observable
            constraintRHS = getUnitRuleRHS(stateQ, U, F, gammaP)
            constraint = truncateComplexObject(constraintRHS - constraintLHS)
            constraint.real[abs(constraint.real) < zero_tol] = 0.0
            constraint.imag[abs(constraint.imag) < zero_tol] = 0.0
            verifySDP = isSemidefinitePositive(constraint)

    return verifySDP

# Updates the observables of stateQ according to the Unit Rule
def applyUnitRule(stateP, stateQ, U, F):
    constraintLHS = getUnitRuleLHS(stateP, F)

    fullDomain, domainIndices = getFullDomain(stateP, F)
    forwardMap = {fullDomain[i]:i for i in range(len(fullDomain))}
    gammaP = fullDomainProjectionExpansion(fullDomain, stateP, forwardMap)

    solveUnitRuleConstraints(constraintLHS, stateQ, fullDomain, domainIndices, gammaP, U, F)

def abstractReasoningStep(state, U, F):
    evolvedAbstractState = abstractStep(state, U, F)
    applyUnitRule(state, evolvedAbstractState, U, F)

    if any(obs is None for obs in evolvedAbstractState.observables):
        # Assume single observable
        _, domainIndices = getFullDomain(state, F)
        evolvedAbstractState.observables[domainIndices[0]] = state.observables[domainIndices[0]]

    assert(verifyUnitRule(state, evolvedAbstractState, U, F))

    return evolvedAbstractState
