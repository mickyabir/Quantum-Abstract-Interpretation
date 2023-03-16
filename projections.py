import numpy as np

from matrixUtil import gramSchmidt

def intersectProjections(projections, tj):
    expandedUnion = None
    for p in projections:
        if expandedUnion is None:
            expandedUnion = p.copy()
        else:
            expandedUnion += p

    fullComplementUnion = len(projections) * np.identity(2 ** len(tj)) - expandedUnion
    complementSupport = getSupport(fullComplementUnion)

    if not complementSupport:
        return np.identity(2 ** len(tj), dtype=complex)
    complementSupportMatrix = getMatrixFromSpan(complementSupport)

    finalSupportMatrix = np.identity(2 ** len(tj)) - complementSupportMatrix

    return finalSupportMatrix

def getSupport(A):
    columnVectors = [A[:, i] for i in range(A.shape[0])]
    gsColumnVectors = gramSchmidt(columnVectors)
    return gsColumnVectors

def getMatrixFromSpan(span):
    dim = span[0].shape[0]
    P_span = np.zeros((dim, dim), dtype=complex)

    for i in range(len(span)):
        P_span[:, i] = span[i]

    return P_span @ P_span.conj().T


