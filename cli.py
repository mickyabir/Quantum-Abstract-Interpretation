from objective import objectiveFunctionMap
from pprint import pprint
from prover import Prover

def session(initialState, ops):
    prover = Prover(initialState, ops)

    displayProj = False
    displayObsv = False

    prevCmd = ''
    while True:
        U, F = prover.getCurrentOp()
        if U is not None and F is not None:
            print(f'F: {F}')
            print('U: ')
            pprint(U)

        userInput = input('>: ').split()

        if len(userInput) == 0:
            cmd = prevCmd
        else:
            cmd = userInput[0]

        prevCmd = cmd

        if cmd in ['n', 'next', 'apply']:
            objectiveFunctionName = None
            if len(userInput) > 1:
                objectiveFunctionName = userInput[1]

            objectiveFunction = objectiveFunctionMap.get(objectiveFunctionName)

            if not prover.apply(objectiveFunction):
                print('Done')
            else:
                if displayProj:
                    prover.proj(F)
                if displayObsv:
                    prover.obsv(F)
        elif cmd in ['b', 'backtrack']:
            _, F = prover.getPrevOp()
            prover.backtrack()
            if displayProj:
                prover.proj(F)
            if displayObsv:
                prover.obsv(F)
        elif cmd in ['o', 'op']:
            U, F = prover.getCurrentOp()
            if U is not None and F is not None:
                print(f'F: {F}')
                print('U: ')
                pprint(U)
        elif cmd in ['p', 'print']:
            prover.print()
        elif cmd in ['v', 'val', 'validate']:
            prover.validate()
        elif cmd in ['h', 'help']:
            print('Help')
        elif cmd in ['d', 'disp', 'display']:
            if len(userInput) == 1:
                displayProj = not displayProj
                displayObsv = not displayObsv
            elif len(userInput) == 2:
                dispType = userInput[1]
                if dispType in ['p', 'proj', 'projection', 'projections']:
                    displayProj = not displayProj
                elif dispType in ['o', 'obsv', 'observable', 'observable']:
                    displayObsv = not displayObsv
        elif cmd in ['exit', 'done']:
            break
