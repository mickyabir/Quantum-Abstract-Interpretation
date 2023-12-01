import numpy as np

from functools import reduce

from matrixUtil import expandUnitary, truncateComplexObject
from projections import intersectProjections

def splitDomainNew(S, F):
    indexList = []
    remainingF = F.copy()
    for i in range(len(S)):
        for f in remainingF:
            if f in S[i]:
                indexList.append(i)
                remainingF.remove(f)

                if not remainingF:
                    return indexList

    return indexList

# Split the domain S into the operating domain set
# Returns indices into S for the active domain
def splitDomain(S, F):
    # return splitDomainNew(S, F)

    # If the domain is the single domain:
    if max(len(s) for s in S) == 1:
        return [S.index([f]) for f in F]
    elif len(S) == 1:
        return [0]

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

def fullDomainObservableExpansion(fullDomain, domainIndices, state, forwardMap=None, backend=np):
    if forwardMap is not None:
        applyForwardMap = lambda S: [forwardMap[si] for si in S]

    M_A = None
    for i in domainIndices:
        if forwardMap is not None:
            F = applyForwardMap(state.S[i])
        else:
            F = state.S[i]
        expandedObservable = expandUnitary(state.observables[i], len(fullDomain), F, backend=backend)

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

    return M_A

def getUnitRuleRHS(state, U, F, gammaP, backend=np):
    fullDomain, domainIndices = getFullDomain(state, F)
    forwardMap = {fullDomain[i]:i for i in range(len(fullDomain))}
    applyForwardMap = lambda S: [forwardMap[si] for si in S]

    M_B = fullDomainObservableExpansion(fullDomain, domainIndices, state, forwardMap, backend=backend)

    mappedF = applyForwardMap(F)
    U_F = expandUnitary(U, len(fullDomain), mappedF)

    return truncateComplexObject(U_F.conj().T @ M_B @ U_F)


