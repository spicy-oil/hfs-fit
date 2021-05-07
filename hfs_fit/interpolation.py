import numpy as np
import matplotlib.pyplot as plt
import hfs_fit.LU as lu

def Gaussian(x, mu, sig, A):
    '''
    Standard unnormalised gussian
    '''
    return A * np.exp(-(x - mu)**2/(2*sig**2))
#%% The coefficients for linear interpolation
def A(x, x_i1, x_i):
    return (x_i1 - x) / (x_i1 - x_i)

def B(x, x_i1, x_i):
    return (x - x_i) / (x_i1 - x_i)
#%%
def lin_interp(x, f_d, x_d):
    '''
    1D linear interpolation.
    Input 1D array of points to estimate, x data and function data arrays.
    Returns estimated function values for given x.
    '''
    f = []
    for pnt in x:
        if (pnt == x_d).tolist().count(1) != 0: #Do not interpolate if value is on data
            f.append(f_d[np.where(x_d == pnt)[0]])
        else:
            temp = (pnt < x_d).tolist().count(0) #Find position index of nearby pnts
            #Nearby x data
            x_i = x_d[temp - 1]
            x_i1 = x_d[temp]
            #Nearby f data
            f_i = f_d[temp - 1]
            f_i1 = f_d[temp]
            f.append(A(pnt, x_i1, x_i) * f_i + B(pnt, x_i1, x_i) * f_i1)
    return f
#%%
def validate_lin_interp():
    '''
    Check if linearly interpolated pnts give rough functional form of sine
    '''
    plt.clf()
    x_d = np.linspace(0, 10, 10) # x data
    f_d = np.sin(x_d) # function data
    x_true = np.linspace(0, 10, 10000) # approximated as a continuum
    f_true = np.sin(x_true) # approximated continuouse true function
    x = np.linspace(0, 10, 500) # pnts to be interpolated
    new_f = lin_interp(x, f_d, x_d) # interpolated pnts
    plt.plot(x_true, f_true, 'r-', label = 'real function')
    plt.plot(x, new_f, 'b-.', label = 'interpolation from data')
    plt.plot(x_d, f_d, 'gx', label = 'data')
    plt.legend()
    plt.grid()
#%%Coefficients required for cubic spline
def C(x, x_i1, x_i):
    return (1./6) * (A(x, x_i1, x_i) ** 3 - A(x, x_i1, x_i)) * (x_i1 - x_i) ** 2
def D(x, x_i1, x_i):
    return (1./6) * (B(x, x_i1, x_i) ** 3 - B(x, x_i1, x_i)) * (x_i1 - x_i) ** 2
def a(x_i, x_i_1):
    return (x_i - x_i_1) / 6
def b(x_i1, x_i_1):
    return (x_i1 - x_i_1) / 3
def c(x_i1, x_i):
    return (x_i1 - x_i) / 6
def d(f_i1, f_i, f_i_1, x_i1, x_i, x_i_1):
    return ((f_i1 - f_i)/(x_i1 - x_i)) - ((f_i - f_i_1)/(x_i - x_i_1))
#%% Cubic spline class
class CS:
    '''
    1D cubic spline class, compute second derivatives only once for convenience.
    '''
    def __init__(self, f_d, x_d):
        self.data_size = len(f_d)
        self.xdata = x_d
        self.fdata = f_d
        self.second_derivatives = self.second_deriv()
        
    def second_deriv(self):
        '''
        Input x and function array datasets, returns second derivatives.
        Constructs matrix equation and solve for second derivatives.
        '''
        matrix = np.zeros((self.data_size, self.data_size))
        vector = np.zeros((self.data_size, 1)) #array of d's defined above to solve second derivs
        #for each second derivative
        for row in range(self.data_size)[1:-1]: #will do top and bottom separately
            matrix[row][row - 1] = a(self.xdata[row], self.xdata[row - 1])
            matrix[row][row] = b(self.xdata[row + 1], self.xdata[row - 1])
            matrix[row][row + 1] = c(self.xdata[row + 1], self.xdata[row])
            vector[row] = d(self.fdata[row + 1], self.fdata[row], self.fdata[row - 1], 
                            self.xdata[row + 1], self.xdata[row], self.xdata[row - 1])
        #Corner values to be 1 AND leave vector end values to be zero, sets BC
        matrix[0][0] = 1
        matrix[-1][-1] = 1
        return lu.solve(matrix, vector).flatten() #Solve using LU decomp
    
    def interp(self, x):
        '''
        Returns cubic spline interpolated values at given x array of pnts.
        '''
        f = []
        
        for pnt in x:
            if (pnt == self.xdata).tolist().count(1) != 0: 
                #Do not interpolate if value is on data
                f.append(self.fdata[np.where(self.xdata == pnt)[0][0]])
            else:
                temp = (pnt < self.xdata).tolist().count(0) # find position of required intervals
                #Nearby x
                x_i = self.xdata[temp - 1]
                x_i1 = self.xdata[temp]
                #Nearby f
                f_i = self.fdata[temp - 1]
                f_i1 = self.fdata[temp]
                #Nearby 2nd derivs
                d2f_i = self.second_derivatives[temp - 1]
                d2f_i1 = self.second_derivatives[temp]
                f.append(A(pnt, x_i1, x_i) * f_i + B(pnt, x_i1, x_i) * f_i1
                         + C(pnt, x_i1, x_i) * d2f_i + D(pnt, x_i1, x_i) * d2f_i1)
        return np.array(f)
#%%
def validate_cubic_sp():
    '''
    Check if cubic spline interpolated pnts give rough functional form of sine
    '''
    plt.clf()
    x_d = np.linspace(0, 10, 8) # x data
    f_d = np.sin(x_d) # function data
    x_true = np.linspace(0, 10, 10000) # approximated as a continuum
    f_true = np.sin(x_true) # approximated continuouse true function
    x = np.linspace(0, 10, 500) # pnts to be interpolated
    spline = CS(f_d, x_d)
    new_f = spline.interp(x) # interpolated pnts
    plt.plot(x_true, f_true, 'r-', label = 'real function')
    plt.plot(x, new_f, 'b-.', label = 'interpolation from data')
    plt.plot(x_d, f_d, 'gx', label = 'data')
    plt.legend()
    plt.grid()


    
