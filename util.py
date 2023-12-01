import numpy as np

from constraintsUtil import fullDomainObservableExpansion, getFullDomain

def computeSubspaceProjection(initialState, finalState):
    fullDomain = {i for i in range(initialState.n)}
    domainIndices = [i for i in range(len(initialState.S))]
    M_A = fullDomainObservableExpansion(fullDomain, domainIndices, initialState)
    M_B = fullDomainObservableExpansion(fullDomain, domainIndices, finalState)

    eigvals, eigvecs = np.linalg.eig(M_B)
    eigvals = np.real(eigvals)
    psi = eigvecs[:, 0]
    a = eigvals[0]
    b = eigvals[1]

    zeroFullObservable = np.zeros(M_A.shape)
    zeroFullObservable[0][0] = 1
    xMatrix = M_A @ zeroFullObservable
    x = np.real(np.trace(xMatrix))

    bound = (x - b) / (a - b)

    return psi, bound
