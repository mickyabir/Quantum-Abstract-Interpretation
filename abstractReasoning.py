import numpy as np

from abstractStep import abstractStep
from constraintsUtil import getFullDomain, getUnitRuleLHS, getUnitRuleRHS, fullDomainProjectionExpansion, fullDomainObservableExpansion
from matrixUtil import isSemidefinitePositive, zero_tol, truncateComplexObject, expandUnitary
from solver import solveUnitRuleConstraints

def validateFinalInequality(initialState, finalState):
    sumInitial = 0
    sumFinal = 0

    for i in range(len(initialState.S)):
        sumInitial += np.trace(initialState.projections[i] @ initialState.observables[i])
        sumFinal += np.trace(finalState.projections[i] @ finalState.observables[i])

    assert(sumInitial <= sumFinal)

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

    # Attempt to fix non-SDP RHS
    if not verifySDP:
        if len(domainIndices) == 1:
            eigvals, eigvecs = np.linalg.eig(constraint)
            lamb = np.diag(eigvals)
            minLamb = np.amin(lamb)
            assert(minLamb < 0)
            lamb = lamb + abs(minLamb) * np.identity(lamb.shape[0])
            newConstraint = eigvecs @ lamb @ np.linalg.inv(eigvecs)

            assert(isSemidefinitePositive(newConstraint))

            applyForwardMap = lambda S: [forwardMap[si] for si in S]
            mappedF = applyForwardMap(F)
            U_F = expandUnitary(U, len(fullDomain), mappedF)

            newObservable = U_F.conj().T @ (newConstraint + constraintLHS) @ U_F
            stateQ.observables[domainIndices[0]] = newObservable

            # Verify new observable
            constraintRHS = getUnitRuleRHS(stateQ, U, F, gammaP)
            constraint = truncateComplexObject(constraintRHS - constraintLHS)
            constraint.real[abs(constraint.real) < zero_tol] = 0.0
            constraint.imag[abs(constraint.imag) < zero_tol] = 0.0
            verifySDP = isSemidefinitePositive(constraint)
#         else:
#             print('Can only fix if M_B = B_i for now')
#             for idx in domainIndices:
#                 stateQ.observables[idx] = np.identity(4)
#                 return True

        if not verifySDP:
            # Try previous obserables
            for idx in domainIndices:
                tempObservable = stateQ.observables[idx]
                stateQ.observables[idx] = stateP.observables[idx]

                # Verify new observable
                constraintRHS = getUnitRuleRHS(stateQ, U, F, gammaP)
                constraint = truncateComplexObject(constraintRHS - constraintLHS)
                constraint.real[abs(constraint.real) < zero_tol] = 0.0
                constraint.imag[abs(constraint.imag) < zero_tol] = 0.0
                verifySDP = isSemidefinitePositive(constraint)

                if verifySDP:
                    return True

                stateQ.observables[idx] = tempObservable

            # Try single identity replacement
            for idx in domainIndices:
                tempObservable = stateQ.observables[idx]
                stateQ.observables[idx] = np.identity(4, dtype=complex)

                # Verify new observable
                constraintRHS = getUnitRuleRHS(stateQ, U, F, gammaP)
                constraint = truncateComplexObject(constraintRHS - constraintLHS)
                constraint.real[abs(constraint.real) < zero_tol] = 0.0
                constraint.imag[abs(constraint.imag) < zero_tol] = 0.0
                verifySDP = isSemidefinitePositive(constraint)

                if verifySDP:
                    return True

                stateQ.observables[idx] = tempObservable

            # Set all working set to identity
            for idx in domainIndices:
                stateQ.observables[idx] = np.identity(4)

    return verifySDP

# Updates the observables of stateQ according to the Unit Rule
def applyUnitRule(stateP, stateQ, U, F):
    fullDomain, domainIndices = getFullDomain(stateP, F)
    forwardMap = {fullDomain[i]:i for i in range(len(fullDomain))}

    if len(F) == 1:
        applyForwardMap = lambda S: [forwardMap[si] for si in S]
        mappedF = applyForwardMap(F)
        U_F = expandUnitary(U, len(fullDomain), mappedF)
        stateQ.observables[domainIndices[0]] = U_F @ stateP.observables[domainIndices[0]] @ U_F
        return

    gammaP = fullDomainProjectionExpansion(fullDomain, stateP, forwardMap)

    constraintLHS = getUnitRuleLHS(stateP, F)

    solveUnitRuleConstraints(constraintLHS, stateQ, fullDomain, domainIndices, gammaP, U, F)

def abstractReasoningStep(state, U, F):
    evolvedAbstractState = abstractStep(state, U, F)
    applyUnitRule(state, evolvedAbstractState, U, F)

    if any(obs is None for obs in evolvedAbstractState.observables):
        # Try to fill in missing observables with previous the observables
        _, domainIndices = getFullDomain(state, F)
        for idx in domainIndices:
            if evolvedAbstractState.observables[idx] is None:
                evolvedAbstractState.observables[idx] = state.observables[idx]

    assert(verifyUnitRule(state, evolvedAbstractState, U, F))

    return evolvedAbstractState
