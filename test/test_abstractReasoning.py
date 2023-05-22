import unittest

import numpy as np

from abstractReasoning import applyUnitRule, verifyUnitRule
from abstractState import AbstractState
from abstractStep import abstractStep
from constraintsUtil import splitDomain

class SplitDomainTest(unittest.TestCase):
    def test_split_domain_one_first(self):
        S = [[0, 1], [1, 2], [2, 3], [3, 4]]
        F = [0]

        indexList = splitDomain(S, F)

        self.assertTrue(indexList == [0])

    def test_split_domain_one_last(self):
        S = [[0, 1], [1, 2], [2, 3], [3, 4]]
        F = [4]

        indexList = splitDomain(S, F)

        self.assertTrue(indexList == [3])

    def test_split_domain_one_middle(self):
        S = [[0, 1], [1, 2], [2, 3], [3, 4]]
        F = [3]

        indexList = splitDomain(S, F)

        self.assertTrue(indexList == [2])

    def test_split_domain_two_separate(self):
        S = [[0, 1], [1, 2], [2, 3], [3, 4]]
        F = [1, 3]

        indexList = splitDomain(S, F)

        self.assertTrue(indexList == [0, 2])

    def test_split_domain_two_together(self):
        S = [[0, 1], [1, 2], [2, 3], [3, 4]]
        F = [2, 3]

        indexList = splitDomain(S, F)

        self.assertTrue(indexList == [2])
        
class VerifyUnitRuleTest(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.S = []

        for i in range(self.n - 1):
            self.S.append([i, i + 1])

        self.initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
        self.initial_obsv = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
        self.state = AbstractState(self.n, self.S, [self.initial_proj for _ in range(self.n - 1)], [self.initial_obsv for _ in range(self.n - 1)])

    def test_verify_unit_rule_example_1(self):
        U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
        F = [0]
        
        proj = np.array([[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]], dtype=complex)
        obsv = np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
        stateQ = AbstractState(self.n, self.S, [proj] + [self.initial_proj for _ in range(self.n - 2)], [obsv] + [self.initial_obsv for _ in range(self.n - 2)])

        self.assertTrue(verifyUnitRule(self.state, stateQ, U, F))
        
    def test_verify_unit_rule_example_2(self):
        U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        F = [0, 1]
        
        projP = np.array([[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]], dtype=complex)
        obsvP = np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
        stateP = AbstractState(self.n, self.S, [projP] + [self.initial_proj for _ in range(self.n - 2)], [obsvP] + [self.initial_obsv for _ in range(self.n - 2)])

        projQ1 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=complex)
        projQ2 = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
        obsvQ = np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
        stateQ = AbstractState(self.n, self.S, [projQ1, projQ2, self.initial_proj], [obsvQ, self.initial_obsv, self.initial_obsv])

        self.assertTrue(verifyUnitRule(stateP, stateQ, U, F))
        
class ApplyUnitRuleTest(unittest.TestCase):
    def setUp(self):
        n = 3
        S = []

        for i in range(n - 1):
            S.append([i, i + 1])

        initial_proj = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
        initial_obsv = 0.25 * np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=complex)
        self.state = AbstractState(n, S, [initial_proj for _ in range(n - 1)], [initial_obsv for _ in range(n - 1)])

    def test_apply_unit_rule(self):
        U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
        F = [0]
        evolvedState = abstractStep(self.state, U, F)
        applyUnitRule(self.state, evolvedState, U, F)

        isSDP = verifyUnitRule(self.state, evolvedState, U, F)

        self.assertTrue(isSDP)

