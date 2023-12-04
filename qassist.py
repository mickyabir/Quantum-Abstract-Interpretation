from abstractState import AbstractState
from cli import session
from prover import Prover

class Program():
    def __init__(self, n, S, initialProjs, initialObsvs, ops):
        self.initialState = AbstractState(n, S, initialProjs, initialObsvs)
        self.ops = ops

def prove(n, generator, proof, lemmas, config=None):
    if config is not None:
        prog = generator(n, config)
    else:
        prog = generator(n)
    prover = Prover(prog.initialState, prog.ops, lemmas)
    proof(prover)

def proveMiddle(n, generatorFront, generatorBack, proof, lemmas, config=None):
    if config is not None:
        front = generatorFront(n, config)
        back = generatorBack(n, config)
    else:
        front = generatorFront(n)
        back = generatorBack(n)
    proverFront = Prover(front.initialState, front.ops, lemmas)
    proverBack = Prover(back.initialState, back.ops, lemmas)
    proof(proverFront, proverBack)

def interactive(n, generator, lemmas, config):
    prog = generator(n, config)
    session(prog.initialState, prog.ops, lemmas, config)

