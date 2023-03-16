import numpy as np

from abstractState import AbstractState
from abstractStep import abstractStep

def generateGHZPaperFull(n):
    S = []

    for i in range(n):
        for j in range(i + 1, n):
            S.append([i, j])

    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(n, S, [initial_proj for _ in range(int(n * (n - 1) / 2))])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = initial_state

    for i in range(0, n - 1):
        nextState = abstractStep(nextState, H, [i])

    nextState = abstractStep(nextState, X, [n - 1])

    for i in range(0, n - 1):
        nextState = abstractStep(nextState, CNOT, [i, n - 1])

    for i in range(0, n):
        nextState = abstractStep(nextState, H, [i])

def generateGHZPaperPartial(n):
    S = []

    for i in range(n - 1):
        S.append([i, i + 1])

    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(n, S, [initial_proj for _ in range(n - 1)])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = initial_state

    for i in range(0, n - 1):
        nextState = abstractStep(nextState, H, [i])

    nextState = abstractStep(nextState, X, [n - 1])

    for i in range(0, n - 1):
        nextState = abstractStep(nextState, CNOT, [i, n - 1])

    for i in range(0, n):
        nextState = abstractStep(nextState, H, [i])

def generateGHZFull(n):
    S = []

    for i in range(n):
        for j in range(i + 1, n):
            S.append([i, j])

    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(n, S, [initial_proj for _ in range(int(n * (n - 1) / 2))])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    nextState = abstractStep(initial_state, H, [0])


    for i in range(1, n):
        nextState = abstractStep(nextState, CNOT, [0, i])

def exampleFromPaper():
    initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    initial_state = AbstractState(3, [[0, 1], [0, 2], [1, 2]], [initial_proj, initial_proj, initial_proj])

    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]], dtype=complex)
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    CNOT = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    print(initial_state)

    state1 = abstractStep(initial_state, H, [0])
    print(state1)

    state2 = abstractStep(state1, H, [1])
    print(state2)

    state3 = abstractStep(state2, X, [2])
    print(state3)

    state4 = abstractStep(state3, CNOT, [1, 2])
    print(state4)

    state5 = abstractStep(state4, CNOT, [0, 2])
    print(state5)

    state6 = abstractStep(state5, H, [0])
    print(state6)

    state7 = abstractStep(state6, H, [1])
    print(state7)

    state8 = abstractStep(state7, H, [2])
    print(state8)

if __name__ == '__main__':
    import sys
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    import pdb
    pdb.set_trace()
    exampleFromPaper()

    # generateGHZPaperPartial(3)
    0/0

    # exampleFromPaper()

    # import time

    # qubitList = [3, 5, 10, 15, 20, 30, 40, 50]

    # for n in qubitList:
    #     prev = time.time()
    #     generateGHZPaperFull(n)
    #     elapsed = time.time() - prev
    #     print(f'{n}: {elapsed}')

    import cProfile, pstats, io
    from pstats import SortKey

    with cProfile.Profile() as pr:
        generateGHZPaperFull(20)

        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

