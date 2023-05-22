from functools import reduce

from matrixUtil import expandUnitary
from projections import intersectProjections

# Split the domain S into the operating domain set
# Returns indices into S for the active domain
def splitDomain(S, F):
    # For now, assume domain of the form (i, i + 1)
    # TODO: Generalize
    # TODO: Optimize if statements/write better
    pairedF = []
    i = 0
    while i < len(F):
        if i < len(F) - 1 and F[i] + 1 == F[i + 1]:
            pairedF.append([F[i], F[i + 1]])
            i += 2
        else:
            pairedF.append([F[i]])
            i += 1

    idxF = 0
    indexList = []
    for i in range(len(S)):
        if idxF >= len(pairedF):
            return indexList

        si = S[i]

        if len(pairedF[idxF]) == 2 and pairedF[idxF] == si:
            indexList.append(i)
            idxF += 1
        elif len(pairedF[idxF]) == 1 and pairedF[idxF][0] in si:
            indexList.append(i)
            idxF += 1

    return indexList

def fullDomainProjectionExpansion(fullDomain, state, forwardMap):
    applyForwardMap = lambda S: [forwardMap[si] for si in S]

    projectionIntersectionList = []
    for i in range(len(state.S)):
        if set(state.S[i]).issubset(set(fullDomain)):
                mappedS = applyForwardMap(state.S[i])
                expandedProjection = expandUnitary(state.projections[i], len(fullDomain), mappedS)
                projectionIntersectionList.append(expandedProjection)
    return intersectProjections(projectionIntersectionList, fullDomain)

def fullDomainObservableExpansion(fullDomain, domainIndices, state, forwardMap, symbolic=False):
    applyForwardMap = lambda S: [forwardMap[si] for si in S]

    M_A = None
    # for i in range(len(domainIndices)):
    for i in domainIndices:
        F = applyForwardMap(state.S[i])
        expandedObservable = expandUnitary(state.observables[i], len(fullDomain), F, symbolic=symbolic)

        if M_A is None:
            M_A = expandedObservable
        else:
            M_A += expandedObservable

    return M_A

def getFullDomain(state, F):
    domainIndices = splitDomain(state.S, F)
    fullDomainList = [set(state.S[i]) for i in domainIndices]
    return list(reduce(lambda a, b: a.union(b), fullDomainList)), domainIndices

def getUnitRuleLHS(state, F):
    fullDomain, domainIndices = getFullDomain(state, F)

    forwardMap = {fullDomain[i]:i for i in range(len(fullDomain))}

    gammaP = fullDomainProjectionExpansion(fullDomain, state, forwardMap)
    M_A = fullDomainObservableExpansion(fullDomain, domainIndices, state, forwardMap)

    return gammaP @ M_A @ gammaP

def getUnitRuleRHS(state, U, F, gammaP, symbolic=False):
    fullDomain, domainIndices = getFullDomain(state, F)
    forwardMap = {fullDomain[i]:i for i in range(len(fullDomain))}
    applyForwardMap = lambda S: [forwardMap[si] for si in S]

    M_B = fullDomainObservableExpansion(fullDomain, domainIndices, state, forwardMap, symbolic=symbolic)

    mappedF = applyForwardMap(F)
    U_F = expandUnitary(U, len(fullDomain), mappedF)

    return gammaP @ U_F.conj().T @ M_B @ U_F @ gammaP


