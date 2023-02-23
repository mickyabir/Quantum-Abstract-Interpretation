import unittest

import numpy as np

class GenerateQubitSwapUnitaryTest(unittest.TestCase):
    def setUp(self):
        from QAI import generateQubitSwapUnitary
        self.generator = generateQubitSwapUnitary

    def test_generate_qubit_swap_unitary_default(self):
        generatedSwapUnitary = self.generator(2, 0, 1)
        concreteUnitary = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        self.assertTrue(np.array_equal(generatedSwapUnitary, concreteUnitary))

    def test_generate_qubit_swap_unitary_random_matrices(self):
        generatedSwapUnitary = self.generator(5, 1, 3)
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        C = np.random.rand(2, 2)
        D = np.random.rand(2, 2)
        E = np.random.rand(2, 2)

        originalMatrix = np.kron(np.kron(np.kron(np.kron(A, B), C), D), E)
        actualMatrix = np.matmul(generatedSwapUnitary, np.matmul(originalMatrix, generatedSwapUnitary))
        expectedMatrix = np.kron(np.kron(np.kron(np.kron(A, D), C), B), E)


        print(actualMatrix)
        print()
        print(expectedMatrix)

        self.assertTrue(np.array_equal(actualMatrix, expectedMatrix))



class ExpandUnitaryTest(unittest.TestCase):
    def test_expand_unitary(self):
        pass
