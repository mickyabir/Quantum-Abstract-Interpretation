import unittest

import numpy as np

from matrixUtil import zero_tol as tol
from unitaryGenerator import generateQubitSwapUnitary, generateQubitLeftRotateUnitary, generateQubitRightRotateUnitary

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


