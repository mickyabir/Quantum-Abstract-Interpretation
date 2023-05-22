import numpy as np

from abstractReasoning import abstractReasoningStep
from abstractState import AbstractState
from abstractStep import abstractStep

def generateGHZPaperPartial(n):
    S = []

    for i in range(n - 1):
        S.append([i, i + 1])

    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_obsv = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
    initial_state = AbstractState(n, S, [initial_proj for _ in range(n - 1)], [initial_obsv for _ in range(n-1)])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = abstractReasoningStep(initial_state, H, [0])

    for i in range(1, n):
        nextState = abstractStep(nextState, CNOT, [0, i])

    import pdb
    pdb.set_trace()

def generateMiller():
    n = 3
    S = []

    for i in range(n - 1):
        S.append([i, i + 1])

    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_obsv = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
    initial_state = AbstractState(n, S, [initial_proj for _ in range(n - 1)], [initial_obsv for _ in range(n-1)])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    T = np.array([[1, 0],[0, np.exp(1j * np.pi / 4)]], dtype=complex)
    TDG = np.array([[1, 0],[0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CNOT10 = np.array([[1, 0, 0, 0],[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)

    nextState = initial_state

    # import pdb
    # pdb.set_trace()

    nextState = abstractReasoningStep(initial_state, CNOT10, [1, 2])
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

    import pdb
    pdb.set_trace()
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

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    # generateGHZPaperPartial(10)
    generateMiller()
