import numpy as np

def pprint_repr(A):
    retStr = ''
    p_tol = 1e-5
    round_tol = 4
    for i in range(len(A)):
        for j in range(len(A[i])):
            realFlag = False
            complexFlag = False
            if abs(A[i][j].real) > p_tol:
                retStr += f'{np.around(A[i][j].real, decimals = round_tol)}'
                realFlag = True
            if abs(A[i][j].imag) > p_tol:
                imagStr = ''
                if realFlag:
                    imagStr += ' + '

                imagStr += f'{np.around(A[i][j].imag, decimals = round_tol)}j'

                retStr += imagStr

                complexFlag = True

            if not realFlag and not complexFlag:
                retStr += f'0'

            retStr += '   '

        retStr += '\n\n'

    return retStr

def pprint(A):
    print(pprint_repr(A))

