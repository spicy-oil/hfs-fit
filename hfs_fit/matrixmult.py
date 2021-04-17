"""Depreciated multiplication module."""
# import numpy as np

# def mult(A, B):
#     '''
#     2darray A & B, where first axis of array A has to equal second axis of array B
#     '''
#     if A.shape[1] != B.shape[0]:
#         print('Incorrect matrix shapes for multiplication.')
#         return
#     r = B.shape[1]
#     c = A.shape[0] #C is row by col matrix in the end
#     C = np.zeros((r*c))
#     temp = 0
#     for row in A:
#         for coloumn in B.T:
#             C[temp] = (row * coloumn).sum()
#             temp += 1
#     C = C.reshape(c, r)
#     return C

# the following code proves that the above method is equivalent to numpy's 
# defualt matrix multiplication

# # create two random arrays
# for _ in range(100):
#     n1, n2, m = np.random.randint(0, high=10, size=3)
#     A = np.random.rand(n1, m)
#     B = np.random.rand(m, n2)
#     np.testing.assert_almost_equal(mult(A, B), A @ B)
