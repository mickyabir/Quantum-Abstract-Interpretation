from abstractState import AbstractState
from cli import session
from prover import Prover

class Program():
    def __init__(self, n, S, initialProjs, initialObsvs, ops):
        self.initialState = AbstractState(n, S, initialProjs, initialObsvs)
        self.ops = ops

def prove(n, generator, proof, config):
    prog = generator(n, config)
    prover = Prover(prog.initialState, prog.ops)
    proof(prover)

def interactive(n, generator, config):
    prog = generator(n, config)
    session(prog.initialState, prog.ops, config)

