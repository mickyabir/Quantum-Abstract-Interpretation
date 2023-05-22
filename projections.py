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
    spanMatrix = None

    for v in span:
        if spanMatrix is None:
            spanMatrix = np.kron(v, v.conj().T).reshape((v.shape[0], v.shape[0]))
        else:
            spanMatrix += np.kron(v, v.conj().T).reshape((v.shape[0], v.shape[0]))

    return spanMatrix
