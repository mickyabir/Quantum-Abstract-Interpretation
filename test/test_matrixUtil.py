import unittest

import numpy as np

from matrixUtil import zero_tol as tol, expandUnitary, gramSchmidt

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

class GramSchmidtTest(unittest.TestCase):
    def test_gram_schmidt_full_one(self):
        v1 = np.array([1, -1, 1], dtype=complex)
        v2 = np.array([1, 0, 1], dtype=complex)
        v3 = np.array([1, 1, 2], dtype=complex)

        vectors = [v1, v2, v3]
        actualVectors = gramSchmidt(vectors)

        expectedV1 = np.array([np.sqrt(3) / 3, -np.sqrt(3) / 3, np.sqrt(3) / 3], dtype=complex)
        expectedV2 = np.array([np.sqrt(6) / 6, np.sqrt(6) / 3, np.sqrt(6) / 6], dtype=complex)
        expectedV3 = np.array([-np.sqrt(2) / 2, 0, np.sqrt(2) / 2], dtype=complex)
        expectedVectors = [expectedV1, expectedV2, expectedV3]

        for i in range(len(expectedVectors)):
            self.assertTrue(np.allclose(actualVectors[i], expectedVectors[i]))

    def test_gram_schmidt_full_two(self):
        v1 = np.array([1, 2, 2], dtype=complex)
        v2 = np.array([-1, 0, 2], dtype=complex)
        v3 = np.array([0, 0, 1], dtype=complex)

        vectors = [v1, v2, v3]
        actualVectors = gramSchmidt(vectors)

        expectedV1 = 1/3 * np.array([1, 2, 2], dtype=complex)
        expectedV2 = 1/3 * np.array([-2, -1, 2], dtype=complex)
        expectedV3 = 1/3 * np.array([2, -2, 1], dtype=complex)
        expectedVectors = [expectedV1, expectedV2, expectedV3]

        for i in range(len(expectedVectors)):
            self.assertTrue(np.allclose(actualVectors[i], expectedVectors[i]))

    def test_gram_schmidt_zero_vector(self):
        v1 = np.array([1, 0, 0], dtype=complex)
        v2 = np.array([0, 1, 0], dtype=complex)
        v3 = np.array([0, 0, 0], dtype=complex)

        vectors = [v1, v2, v3]

        actualVectors = gramSchmidt(vectors)

        expectedV1 = np.array([1, 0, 0], dtype=complex)
        expectedV2 = np.array([0, 1, 0], dtype=complex)
        expectedV3 = np.array([0, 0, 0], dtype=complex)

        self.assertTrue(len(actualVectors) == 2)
        self.assertTrue(np.allclose(actualVectors[0], expectedV1))
        self.assertTrue(np.allclose(actualVectors[1], expectedV2))
