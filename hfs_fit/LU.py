import numpy as np
import hfs_fit.matrixmult as mm

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
    return mm.mult(inverse(A), b)
#%%
def validate():
    '''
    Using the LU() funtion to decompose given matrix, and multiplying result L & U.
    '''
    A = np.array([[  2,   1,   0,   0,   0],
                  [  3,   8,   4,   0,   0],
                  [  0,   9,  20,  10,   0],
                  [  0,   0,  22,  51, -25],
                  [  0,   0,   0, -55,  60]])
    decomposed = LU(A) #LU decomposition of A
    recombined = mm.mult(decomposed[0], decomposed[1])
    if np.abs((recombined - A).sum()) <= 1e-6: #LU should be same as original A
        print('Decomposition is correct.')
    A_1 = inverse(A)
    identity = mm.mult(A_1, A) #Multiplying A with its inverse
    if np.abs((identity - np.identity(len(A))).sum()) <= 1e-6: #comparing to identity matrix
        print('Inversion is correct and hence forward & backward substitution is correct.')
        
#%%
def LUk(A):
    '''
    Input square matrix (2d numpy array), returns lower L and upper triang U matrices.
    BEFORE ANY OPTIMISATION
    '''
    N = len(A[0]) # N x N matrix
    L = np.identity(N) #Crout's algorithm requires 1 along diagonal of L
    U = np.zeros((N, N))
    for diag in range(N): #For every diagonal
        for j in range(N): #Solving for corresponding row in U 
            sum_ku = 0 #Sum over k
            for k in range(diag):
                sum_ku += L[diag][k] * U[k][j]
            U[diag][j] = A[diag][j] - sum_ku
        for i in range(N): #Solving for corresponding column in L
            sum_kl = 0 #Sum over k
            for k in range(diag):
                sum_kl += L[i][k] * U[k][diag]
            L[i][diag] = (A[i][diag] - sum_kl) / U[diag][diag]      
    return L, U

#%%
def LU3(A):
    '''
    Input 3x3 matrix (2d numpy array), returns lower L and upper triang U matrices.
    THIS WAS USED TO UNDERSTAND THE ALGORITHM.
    '''
    N = len(A[0]) # N x N matrix
    L = np.identity(N) #Crout's algorithm requires 1 along diagonal of L
    U = np.zeros((N, N))
    ############
    # for first diagonal
    ############
    diag = 0
    U[diag][diag] = A[diag][diag] / L[diag][diag] #u_00 = a_00 / l_00
    #now can calculate first column of L
    for i in range(N): #Each row
        L[i][diag] = A[i][diag] / U[diag][diag]
    #now can calculate first row of U
    for j in range(N):
        U[diag][j] = A[diag][j] / L[diag][diag] #L[diag][diag] should always be 1
    ############
    # for second diagonal
    ############
    diag = 1
    #U[diag][diag] = A[diag][diag] - L[diag][0] * U[0][diag]
    #now second row of U
    for j in range(N):
        U[diag][j] = A[diag][j] - L[diag][0] * U[0][j]
    #for second column of L
    for i in range(N):
        L[i][diag] = (A[i][diag] - L[i][0] * U[0][diag]) / U[diag][diag]
    #for diag in range(N): #index of a diagonal in N x N matrix i.e. A[diag][diag]
    ############
    # for thrid diagonal
    ############
    diag = 2
    #now third row of U
    for j in range(N):
        U[diag][j] = A[diag][j] - L[diag][0] * U[0][j] - L[diag][1] * U[1][j]
    #thirdcolumn of L
    for i in range(N):
        L[i][diag] = (A[i][diag] - L[i][0] * U[0][diag] - L[i][1] * U[1][diag]) / U[diag][diag]
    return L, U
            
