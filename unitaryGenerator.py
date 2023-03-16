import numpy as np

def generateQubitRightRotateUnitary(n, i):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    rotateUnitary = np.identity(2 ** n)
    for k in range(2 ** n):
        binaryStringK = list(getBin(k, n))

        leftStringK = binaryStringK[0:i + 1]
        rightStringK = binaryStringK[i + 1:len(binaryStringK)]
        leftStringK = [leftStringK[-1]] + leftStringK[0:len(leftStringK) - 1]
        intK = int(''.join(leftStringK + rightStringK), 2)

        rotateUnitary[k][k] = 0
        rotateUnitary[k][intK] = 1

    return rotateUnitary

def generateQubitLeftRotateUnitary(n, i):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    rotateUnitary = np.identity(2 ** n)
    for k in range(2 ** n):
        binaryStringK = list(getBin(k, n))

        leftStringK = binaryStringK[0:i + 1]
        rightStringK = binaryStringK[i + 1:len(binaryStringK)]
        leftStringK = leftStringK[1:len(leftStringK)] + [leftStringK[0]]
        intK = int(''.join(leftStringK + rightStringK), 2)

        rotateUnitary[k][k] = 0
        rotateUnitary[k][intK] = 1

    return rotateUnitary

def generateQubitSwapUnitary(n, i, j):
    if n == 1:
        return np.identity(2)

    getBin = lambda x, n: format(x, 'b').zfill(n)

    swapUnitary = np.identity(2 ** n)

    for k in range(2 ** n):
        binaryStringK = list(getBin(k, n))
        tmp = binaryStringK[i]
        binaryStringK[i] = binaryStringK[j]
        binaryStringK[j] = tmp
        binaryStringK = ''.join(binaryStringK)
        intK = int(binaryStringK, 2)

        swapUnitary[k][k] = 0
        swapUnitary[k][intK] = 1

    return swapUnitary
