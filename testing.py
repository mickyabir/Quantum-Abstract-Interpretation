import unittest

import numpy as np

from QAI import tol, generateQubitSwapUnitary, generateQubitSwapFrontUnitary, generateQubitRightRotateUnitary, generateQubitLeftRotateUnitary, expandUnitary

class GenerateQubitSwapUnitaryTest(unittest.TestCase):
    def test_generate_qubit_swap_unitary_default(self):
        generatedSwapUnitary = generateQubitSwapUnitary(2, 0, 1)
        concreteUnitary = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        self.assertTrue(np.array_equal(generatedSwapUnitary, concreteUnitary))

    def test_generate_qubit_swap_unitary_random_matrices_middle(self):
        generatedSwapUnitary = generateQubitSwapUnitary(5, 1, 3)
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        C = np.random.rand(2, 2)
        D = np.random.rand(2, 2)
        E = np.random.rand(2, 2)

        originalMatrix = np.kron(np.kron(np.kron(np.kron(A, B), C), D), E)
        actualMatrix = np.matmul(generatedSwapUnitary, np.matmul(originalMatrix, generatedSwapUnitary))
        expectedMatrix = np.kron(np.kron(np.kron(np.kron(A, D), C), B), E)

        self.assertTrue(np.allclose(actualMatrix, expectedMatrix, atol=tol))

    def test_generate_qubit_swap_unitary_random_matrices_edges(self):
        generatedSwapUnitary = generateQubitSwapUnitary(5, 0, 4)
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        C = np.random.rand(2, 2)
        D = np.random.rand(2, 2)
        E = np.random.rand(2, 2)

        originalMatrix = np.kron(np.kron(np.kron(np.kron(A, B), C), D), E)
        actualMatrix = np.matmul(generatedSwapUnitary, np.matmul(originalMatrix, generatedSwapUnitary))
        expectedMatrix = np.kron(np.kron(np.kron(np.kron(E, B), C), D), A)

        self.assertTrue(np.allclose(actualMatrix, expectedMatrix, atol=tol))

class GenerateQubitSwapFrontUnitaryTest(unittest.TestCase):
    def test_generate_qubit_front_swap_unitary_default(self):
        generatedSwapFrontUnitary = generateQubitSwapFrontUnitary(3, [1, 2])
        concreteUnitary = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]], dtype=complex)

        # self.assertTrue(np.array_equal(generatedSwapUnitary, concreteUnitary))

class GenerateQubitRotateUnitary(unittest.TestCase):
    def test_generate_qubit_rotate_unitary_full_vector(self):
        generatedRightRotateUnitary = generateQubitRightRotateUnitary(3, 2)
        V = np.random.rand(8)
        rotateV = generatedRightRotateUnitary @ V

        self.assertTrue(rotateV[1] == V[4])
        self.assertTrue(rotateV[2] == V[1])
        self.assertTrue(rotateV[3] == V[5])
        self.assertTrue(rotateV[4] == V[2])
        self.assertTrue(rotateV[5] == V[6])
        self.assertTrue(rotateV[6] == V[3])

    def test_generate_qubit_rotate_unitary_half_vector(self):
        generatedRightRotateUnitary = generateQubitRightRotateUnitary(4, 2)
        V = np.random.rand(16)
        rotateV = generatedRightRotateUnitary @ V

        self.assertTrue(rotateV[1] == V[1])
        self.assertTrue(rotateV[2] == V[8])
        self.assertTrue(rotateV[3] == V[9])
        self.assertTrue(rotateV[4] == V[2])
        self.assertTrue(rotateV[5] == V[3])
        self.assertTrue(rotateV[6] == V[10])
        self.assertTrue(rotateV[7] == V[11])

        self.assertTrue(rotateV[8] == V[4])
        self.assertTrue(rotateV[9] == V[5])
        self.assertTrue(rotateV[10] == V[12])
        self.assertTrue(rotateV[11] == V[13])
        self.assertTrue(rotateV[12] == V[6])
        self.assertTrue(rotateV[13] == V[7])
        self.assertTrue(rotateV[14] == V[14])
        self.assertTrue(rotateV[15] == V[15])

    def test_generate_qubit_rotate_unitary_three_qubits(self):
        generatedRightRotateUnitary = generateQubitRightRotateUnitary(3, 2)
        generatedLeftRotateUnitary = generateQubitLeftRotateUnitary(3, 2)
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        C = np.random.rand(2, 2)

        originalMatrix = np.kron(np.kron(A, B), C)
        actualMatrix = np.matmul(generatedLeftRotateUnitary, np.matmul(originalMatrix, generatedRightRotateUnitary))
        expectedMatrix = np.kron(np.kron(C, A), B)

        self.assertTrue(np.allclose(actualMatrix, expectedMatrix, atol=tol))

    def test_generate_qubit_rotate_unitary_zero(self):
        generatedRightRotateUnitary = generateQubitRightRotateUnitary(5, 0)
        generatedLeftRotateUnitary = generateQubitLeftRotateUnitary(5, 0)
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        C = np.random.rand(2, 2)
        D = np.random.rand(2, 2)
        E = np.random.rand(2, 2)

        originalMatrix = np.kron(np.kron(np.kron(np.kron(A, B), C), D), E)
        actualMatrix = np.matmul(generatedLeftRotateUnitary, np.matmul(originalMatrix, generatedRightRotateUnitary))
        expectedMatrix = np.kron(np.kron(np.kron(np.kron(A, B), C), D), E)

        self.assertTrue(np.allclose(actualMatrix, expectedMatrix, atol=tol))

    def test_generate_qubit_rotate_unitary_one(self):
        generatedRightRotateUnitary = generateQubitRightRotateUnitary(5, 1)
        generatedLeftRotateUnitary = generateQubitLeftRotateUnitary(5, 1)
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        C = np.random.rand(2, 2)
        D = np.random.rand(2, 2)
        E = np.random.rand(2, 2)

        originalMatrix = np.kron(np.kron(np.kron(np.kron(A, B), C), D), E)
        actualMatrix = np.matmul(generatedLeftRotateUnitary, np.matmul(originalMatrix, generatedRightRotateUnitary))
        expectedMatrix = np.kron(np.kron(np.kron(np.kron(B, A), C), D), E)

        self.assertTrue(np.allclose(actualMatrix, expectedMatrix, atol=tol))

    def test_generate_qubit_rotate_unitary_middle(self):
        generatedRightRotateUnitary = generateQubitRightRotateUnitary(5, 2)
        generatedLeftRotateUnitary = generateQubitLeftRotateUnitary(5, 2)
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        C = np.random.rand(2, 2)
        D = np.random.rand(2, 2)
        E = np.random.rand(2, 2)

        originalMatrix = np.kron(np.kron(np.kron(np.kron(A, B), C), D), E)
        # actualMatrix = np.matmul(generatedLeftRotateUnitary, np.matmul(originalMatrix, generatedRightRotateUnitary))
        actualMatrix = generatedLeftRotateUnitary @ originalMatrix @ generatedRightRotateUnitary
        expectedMatrix = np.kron(np.kron(np.kron(np.kron(C, A), B), D), E)

        self.assertTrue(np.allclose(actualMatrix, expectedMatrix, atol=tol))

class ExpandUnitaryTest(unittest.TestCase):
    def test_expand_unitary_small_edges(self):
        U = np.array([[1, 2], [3, 4]])
        V = np.array([[5, 6], [7, 8]])
        I = np.identity(2)
        tensorUV = np.kron(U, V)

        expandedUV = expandUnitary(tensorUV, 3, [0, 2])
        actualMatrix = np.kron(U, np.kron(I, V))

        self.assertTrue(np.allclose(expandedUV, actualMatrix))

    def test_expand_unitary_medium_edges(self):
        U = np.array([[1, 2], [3, 4]])
        V = np.array([[5, 6], [7, 8]])
        I = np.identity(2)
        tensorUV = np.kron(U, V)

        expandedUV = expandUnitary(tensorUV, 5, [0, 4])
        actualMatrix = np.kron(U, np.kron(I, np.kron(I, np.kron(I, V))))

        self.assertTrue(np.allclose(expandedUV, actualMatrix))

    def test_expand_unitary_small_middle(self):
        U = np.array([[1, 2], [3, 4]])
        V = np.array([[5, 6], [7, 8]])
        I = np.identity(2)
        tensorUV = np.kron(U, V)

        actualMatrix = np.kron(I, np.kron(U, V))
        expandedUV = expandUnitary(tensorUV, 3, [1, 2])

        self.assertTrue(np.allclose(expandedUV, actualMatrix))


    def test_expand_unitary(self):
        U = np.random.rand(2, 2)
        V = np.random.rand(2, 2)
        I = np.identity(2)
        tensorUV = np.kron(U, V)

        expandedUV = expandUnitary(tensorUV, 6, [1, 3])
        actualMatrix = np.kron(I, np.kron(U, np.kron(I, np.kron(V, np.kron(I, I)))))

        self.assertTrue(np.allclose(expandedUV, actualMatrix))

