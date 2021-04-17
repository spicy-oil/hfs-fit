import numpy as np

#%%
def LU(A):
    '''
    Input square matrix (2d numpy array), returns lower L and upper triang U matrices and determinant
    '''
    N = len(A[0]) # N x N matrix
    L = np.identity(N) #Crout's algorithm requires 1 along diagonal of L
    U = np.zeros((N, N))
    for diag in range(N): #For every diagonal
        for j in np.arange(diag, N): #Solving for corresponding row in U ignoring zero entries
            U[diag][j] = A[diag][j] - (L[diag][:diag] * U[:, j][:diag]).sum() #Array sum instead of loop
        for i in np.arange(diag, N): #Solving for corresponding column in L ignoring zero entries
            L[i][diag] = (A[i][diag] - (L[i][:diag] * U[:, diag][:diag]).sum()) / U[diag][diag]    #Array sum instead of loop   
    return L, U, np.diag(U).sum()

#%%
def forward(L, b):
    '''
    Input lower triang matrix L to solve equation Ly = b where y and b are vectors.
    Not generalised for y and b being matrices.
    '''
    N = len(L)
    y = np.zeros(b.shape) #zeroes to fill-in
    for row in range(N): #for each y
        #subtract sum of products of required L & y elements from the row of b
        y[row] = b[row] - (L[row][:row] * y[:, 0][:row]).sum() 
    return y
    
#%%
def backward(U, y):
    '''
    Input upper triang matrix U to solve equation Ux = y where x and y are vectors.
    Not generalised for y and b being matrices.
    '''
    N = len(U)
    x = np.zeros(y.shape)
    for row in range(N)[::-1]: #reverse row indexing schedule
        #subtract sum of products of required U & x elements from the row of y
        x[row] = (y[row] - (U[row][row + 1:] * x[:, 0][row + 1:]).sum()) / U[row][row]
    return x
#%%
def inverse(A):
    '''
    Returns inverse of matrix A using LU and forward-backward substitution.
    '''
    N = len(A)
    decomposed = LU(A) #decomposing A
    b = np.identity(N)
    x = []
    for column in range(N): #Solve each column separately
        bc = b[:, column].reshape(N,1) #Require 2D array to be column vector
        xc = backward(decomposed[1], forward(decomposed[0], bc)) #Solving for x
        x.append(xc) #Appending to form a list of all columns of inverse
    return np.hstack(x) #Stacking the columns to form inverse
#%%
def solve(A, b):
    '''
    Solve matrix equation Ax = b
    '''
    return inverse(A) @ b
