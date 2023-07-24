import numpy as np

from abstractReasoning import abstractReasoningStep, validateFinalInequality
from abstractState import AbstractState

def generateMiller():
    n = 3

    # S = []
    # for i in range(n - 1):
    #   S.append([i, i + 1])

    # initialProj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    # initialObsv = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
    # initialState = AbstractState(n, S, [initialProj for _ in range(n - 1)], [initialObsv for _ in range(n-1)])

    # FULL ABSTRACT DOMAIN
    S = [{0, 1, 2}]
    zero = np.array([[1], [0]], dtype=complex)
    zzz = np.kron(zero, np.kron(zero, zero))
    initialProj = np.kron(zzz, zzz.conj().T)
    plus = 1 / np.sqrt(2) * np.array([[1], [1]], dtype=complex)
    ppp = np.kron(plus, np.kron(plus, plus))
    initialObsv = np.kron(ppp, ppp.conj().T)
    initialState = AbstractState(n, S, [initialProj], [initialObsv])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    T = np.array([[1, 0],[0, np.exp(1j * np.pi / 4)]], dtype=complex)
    TDG = np.array([[1, 0],[0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CNOT10 = np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)

    nextState = initialState

    import pdb
    pdb.set_trace()

    nextState = abstractReasoningStep(initialState, CNOT10, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [1, 0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [1, 0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, TDG, [0])
    print(nextState)
    nextState = abstractReasoningStep(nextState, T, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [0, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [1, 2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT, [0, 1])
    print(nextState)
    nextState = abstractReasoningStep(nextState, H, [2])
    print(nextState)
    nextState = abstractReasoningStep(nextState, CNOT10, [1, 2])
    print(nextState)