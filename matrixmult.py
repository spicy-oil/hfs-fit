import numpy as np

def mult(A, B):
    '''
    2darray A & B, where first axis of array A has to equal second axis of array B
    '''
    if A.shape[1] != B.shape[0]:
        print('Incorrect matrix shapes for multiplication.')
        return
    r = B.shape[1]
    c = A.shape[0] #C is row by col matrix in the end
    C = np.zeros((r*c))
    temp = 0
    for row in A:
        for coloumn in B.T:
            C[temp] = (row * coloumn).sum()
            temp += 1
    C = C.reshape(c, r)
    return C
