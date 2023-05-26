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
    return isSemidefinitePositive(constraint)

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
    assert(verifyUnitRule(state, evolvedAbstractState, U, F))

    return evolvedAbstractState
