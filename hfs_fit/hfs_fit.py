import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from astropy.modeling.models import Voigt1D
import copy as cp
import pandas as pd

import hfs_fit.interpolation as interp
import hfs_fit.relInt as ri


#Change in wavenumber from hfs splitting (of a fine-structure level) is given by 
def K(F, J, I):
    return F * (F + 1) - J * (J + 1) - I * (I + 1)

def dE(A, B, K, F, J, I):
    '''
    B should be zero when I or J is 0 or 1/2 to avoid division by zero
    '''
    if J != .5 and J != 0.:
        return 0.5 * A * K + B * ( (3. / 4 ) * K * (K + 1) - J * (J + 1) * I * (I + 1)) / (2 * I * (2*I - 1) * J * (2*J - 1))
    else:
        return 0.5 * A * K

#Definition and use of the saturation parameter
def satFit(fit, sat):
    return fit * np.exp(- sat * fit)


def get_user_wavenumber():
    """Get the users input for hfs.WNRange.
    
    # TODO make this something the user passes into the class instanciation
    rathing than insiting on dynamic input.

    # TODO add some descriptions as to WTF each of these options are
    rather than assuming omniscience from the user

    returns:
        upperLevel
        lowerLevel
        start_wn
        end_wn

    """
    print('Upper Level Label: ')
    upperLevel = str(input())
    print('Lower Level Label: ')
    lowerLevel = str(input())
    print('Starting wavenumber (/cm): ')
    start_wn = float(input())
    print('End wavenumber (/cm): ')
    end_wn = float(input())
    return upperLevel, lowerLevel, start_wn, end_wn


def get_user_noise():
    """Get the users input for hfs.Noise.
    
    # TODO all the todos of get_user_wavenumber here aswell

    returns:
        start_wn
        end_wn
        
    """
    print('Noise estimation starting wavenumber (/cm): ')
    start_wn = float(input())
    print('Noise estimation end wavenumber (/cm): ')
    end_wn = float(input())
    return start_wn, end_wn


def get_user_levels():
    """Get the users input for hfs.SetJNoLevList.
    
    # TODO all the todos of get_user_wavenumber here aswell

    returns:
        start_wn
        end_wn
        
    """
    print('Upper level J value: ')
    upperJ = float(input())
    print('Lower level J value: ')
    lowerJ = float(input())    
    return upperJ, lowerJ


class hfs:
    def __init__(self, dataASCII = 'spectrum.txt', fitLog = 'fitLog.xlsx', nuclearSpin = 3.5):
        '''
        input strings can be directories to the files.
        dataASCII is the string of directory to asc file of spectral data
        (CSV, first column is wavenumber, second column is intensity)
        '''
        if fitLog != None:
            self.fitLogName = fitLog
            self.fitLog = pd.read_excel(self.fitLogName, index_col = 0)
        else:
            self.fitLogName = None
            self.fitLog = None
        self.I = nuclearSpin
        print('Loading spectrum...')
        self.dataFile = dataASCII
        if isinstance(dataASCII, str) == True:
            self.data = np.loadtxt(self.dataFile, delimiter = ',')
        else:
            print('Please use an ASCII file for the spectrum.')
            return
        print('Done')

        self.paramsUnits = ['mK','mK','mK','mK','mK','mK','arb.','/cm','arb.']
        self.paramsNames  = ['A_u', 'A_l', 'B_u', 'B_l', 'G_w', 'L_w', 'Area Parameter', 'CoG Wavenumber', 'Saturation Parameter']

    def PlotSpec(self, wn = None):
        '''
        Plots the spectrum, specifying a wavenumber will plot a +- 1/cm range around it.
        '''
        plt.close(0)
        plt.figure(0)
        plt.plot(self.data[:, 0], self.data[:, 1], label = 'Spectrum')        
        plt.xlabel(r'Wavenumber (cm$^{-1}$)')
        plt.ylabel('Intensity (arb.)')
        plt.title('Spectrum ' + self.dataFile)
        if wn != None:
            plt.xlim(wn - 1, wn + 1)
        plt.grid()
        plt.show()

    def NewFit(self):
        '''
        Call this whenever you want to fit a new line.
        Resets everything (J, A, B, Gw, etc.) so make sure you save the previous fit.
        Have the wn range for the line and wn range for the noise ready.
        '''
        self.SetJNoLevList()
        self.firstFit = True
        self.Hold()
        self.ParamsGuess()
        self.lineDerivSum = self.DerivSum()

    def WNRange(self, nInterp = 1):
        '''
        set wavenumber range for fitting
        nInterp is number of points of even spacing between actual data to interpolate (cubic spline), set 1 for no interpolation
        note nInterp will chagne the value from DerivSum().
        '''
        self.upperLevel, self.lowerLevel, self.start_wn, self.end_wn = get_user_wavenumber()
        self.FitDone(self.upperLevel, self.lowerLevel)
        line = np.array([d for d in self.data if d[0] >= self.start_wn and d[0] <= self.end_wn])
        self.line = cp.copy(line) #original, non-normalised non-interpolated data
        self.normFactor = line[:, 1].max()
        line[:, 1] /= self.normFactor #normalise line by amplitude 
        #number of points between two neghibouring data points to interpolate
        self.w = np.linspace(line[:, 0][0], line[:, 0][-1], nInterp * (len(line[:, 0])) - nInterp + 1)
        self.spline = interp.CS(line[:, 1], line[:, 0])
 
        #------------------------------------------------------
        #This is the intensity of data interpolated and fitted.
        self.i = self.spline.interp(self.w) #small i attribute is the interpolated instensity
        #------------------------------------------------------

        self.N = self.w.size #number of points after interpolation
        self.Noise()
        temp = np.fft.fft(self.i)
        for i, val in enumerate(temp):
            if np.abs(val) < 0.01 * np.max(np.abs(temp)):
                self.icut = i - 1 # -1 seems to give the best index for apodisation, can change this in plot anyway.
                return

    def Noise(self):
        '''
        Estimates noise and SNR.
        With given inputs.
        '''
        start_wn, end_wn = get_user_noise()
        line = np.array([d for d in self.data if d[0] >= start_wn and d[0] <= end_wn])
        line[:, 1] /= self.normFactor
        self.SNR = 1. / np.std(line[:, 1], ddof = 1)
        
        
    def FitDone(self, u, l):
        '''
        -Check if fit already been done
        -Inputs in strings for s (spectrum), u (upper lev), l (lower lev)
        '''
        for i, r in self.fitLog.iterrows():
            if r[0] == self.dataFile:
                if r[1] == u:
                    if r[2] == l:
                        print('Fit of line already done')

    def DerivSum(self):
        '''
        -Used to find how wiggly a line is within self.WNRange(), 
        -Usually the more wiggly a line is the better parameters are constrained.
        -Essentially the sum of the magnitude of differences in intensity between 
        neighbouring points divided by resolution.
        '''
        #Normalise x axis
        w = self.w - self.w[0]
        w /= w[-1]
        i = self.FitLine(self.paramsGuess)
        d = np.gradient(i, w)
        return np.abs(d).sum()
        

    def SetJNoLevList(self):
        '''
        Sets J values, calculate all transitions and list in terms of F values. Relative intensity calculation.
        '''
        self.upperJ, self.lowerJ = get_user_levels() 
        self.AllowedTransitions()
        self.Swing()
        self.WNRange()

    def AllowedTransitions(self):
        '''
        Calculates relative intensities of allowed HFS transitions of the fine structure transition.
        '''
        self.listF = ri.AllowedTransitions(self.I, self.upperJ, self.lowerJ) #list of [upperF, lowerF] transitions
        relIntensities = []
        for f in self.listF:
            relIntensities.append(ri.RelIntensity(self.I, self.upperJ, self.lowerJ, f[0], f[1]))
        self.relIntensities = np.array(relIntensities) #intensity ratios for each transition

    def ParamsGuess(self, p = [0, 0, 0, 0, 0.150, 0.005, -1, -1, 0]):
        '''
        Sets all parameter guesses.
        As, Bs, Gw and Lw are input in /cm.
        p = [AUpperGuess, ALowerGuess, BUpperGuess, BLowerGuess, sGaussianGuess, gCauchyGuess, areaGuess, cogwnGuess, saturationGuess]
        our results are usally doppler dominated so expect lorentzian width to be very small
        CoG wavenumber and area guesses are calculated automatically based on the selected wavenumber range and line profiles
        '''
        #Obtain guesses for As and Bs
        temp = np.array(p, dtype = 'float64')
        temp[7] = (self.w * self.i).sum() / self.i.sum()
        area = np.trapz(self.i, self.w)
        temp[6] = area
        self.paramsGuess = temp
        #Guess B to be zero, since they tend to be small.
        self.paramsGuess[2] = 0
        self.paramsGuess[3] = 0
        print('Current rms = ' + str(self.Residual(self.paramsGuess)))

    def ApoFit(self, fit):
        '''
        Given a fit of Voigt functions sum, apodise the interferogram (phase included, using the complex amplitude)
        of the fitted spectrum from where the observed interferogram first reaches zero.
        this needs to be done as the mirrors moved a finite distance, apodising the interferogram and caused ringing of lines 
        '''
        I_fit = np.fft.fft(fit)
        self.I_apo_fit = cp.copy(I_fit)
        self.I_apo_fit[self.icut:-self.icut] = 0 #apodise real and complex part
        S_apo_fit = np.fft.ifft(self.I_apo_fit)
        return np.real(S_apo_fit)

    def PlotApo(self):
        '''
        Shows line and guess line in interferogram domain, before and after apodisation of set self.icut
        '''
        #temp = self.FitLine(self.paramsGuess) #generates apodised and non-apodised fits
        I_guess = np.fft.fft(self.noApoFit)
        I_data = np.fft.fft(self.i)
        plt.plot(self.w, np.abs(I_data), '-ro', label = 'data')
        plt.plot(self.w, np.abs(I_guess), '-bo', label = 'fit')
        plt.plot(self.w, np.abs(self.I_apo_fit), '-ko', label = 'apodised fit')
        plt.legend()
        plt.grid()
        plt.show()

    def Hold(self, b = [0, 0, 0, 0, 0, 0, 0, 0, 1]):
        '''
        Set which paramter to hold in self.paramGuess, 1 to hold, 0 to not hold.
        same indices as self.paramsGuess
        Can just turn each on and off in the fitting plot.
        '''
        self.boo = b

    def Swing(self, swing = [1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-3, 5e-2, .1e-2, .1]):
        '''
        Sets Gaussian standard deviations for the jump distribution for each parameter.
        As, Bs, Gw and Lw are input in /cm.
        Same indices as self.paramsGuess
        This is reset after every call of NewFit()
        '''
        self.jumpWidths = swing

    def Residual(self, params):
        '''
        rms, to be minimised, defined as the standard deviation of offsets between each observed intensity and model intensity.
        '''
        line_fit = self.FitLine(params)
        return np.sqrt( ((line_fit - self.i)**2).sum() / (len(self.w) - len(self.boo) + np.sum(self.boo)) )

    def FitLine(self, params):
        '''
        #Calulates the fitted line at given params and wavenumbers
        '''
        upperFWavenumbers = np.zeros_like(self.relIntensities)
        for i, F in enumerate(self.listF[:, 0]):
            upperFWavenumbers[i] += dE(params[0], params[2], K(F, self.upperJ, self.I), F, self.upperJ, self.I)

        lowerFWavenumbers = np.zeros_like(self.relIntensities)
        for i, F in enumerate(self.listF[:, 1]):
            lowerFWavenumbers[i] += dE(params[1], params[3], K(F, self.lowerJ, self.I), F, self.lowerJ, self.I)

        self.hfsLines = upperFWavenumbers - lowerFWavenumbers + params[7]
        line_fit = np.zeros_like(self.w)
        self.components = []
        for i, v in enumerate(self.relIntensities):
            voigt = Voigt1D(self.hfsLines[i], v, params[5], params[4])
            line_fit += voigt(self.w) / (np.pi * params[5] / 2) #normalisation of area
            self.components.append(voigt(self.w) / (np.pi * params[5] / 2))
        line_fit = line_fit * params[6] / np.sum(self.relIntensities)
        self.components = np.array(self.components) * params[6] / np.sum(self.relIntensities)
        line_fit = satFit(line_fit, params[8])
        self.components = satFit(self.components, params[8])
        self.noApoFit = cp.copy(line_fit)
        line_fit = self.ApoFit(line_fit)
        return line_fit

    def TrialParam(self, params):
        '''
        Second array is a boolean array to identify which parametersare kept constant
        '''
        for i, b in enumerate(self.boo):
            if b == 0: #if False in being kept constant
                newTempParam = np.random.normal(params[i], self.jumpWidths[i], 1)[0]
                if i >= 8:
                    params[i] = abs(newTempParam) #saturation parameter cannot be negative...
                elif i == 4:
                    params[i] = abs(newTempParam) #Gw can't be negative
                elif i == 5:
                    params[i] = abs(newTempParam) #Lw can't be negative
                else:
                    params[i] = newTempParam
        return params

    def Anneal(self):
        '''
        The simulated annealing algorithim, temperature parameter is multiplied by, say, .98 every iteration
        '''
        step = 1
        currentParams = self.paramsGuess
        currentResidual = self.Residual(params = self.paramsGuess)
        T = currentResidual * .2
        #In case of hopping into a worse local minimum, record what was best!
        bestParams = cp.copy(currentParams)
        bestResidual = cp.copy(currentResidual)
        while (T - 1e-10) > 0:
            step += 1
            newParams = self.TrialParam(cp.copy(currentParams))
            newResidual = self.Residual(params = newParams)
            residualChange = newResidual - currentResidual
            if residualChange <= 0:
                print('Residual decreased! Current residual = ' + str(newResidual) + ' step = ' + str(step))
                currentParams = cp.copy(newParams)
                currentResidual = cp.copy(newResidual)
                if currentResidual < bestResidual:
                    bestParams = cp.copy(currentParams)
                    bestResidual = cp.copy(currentResidual)
            else:
                boltzProb = np.exp(-residualChange/T)
                if np.random.uniform() <= boltzProb:
                    print('Randomly Excited!')
                    currentParams = cp.copy(newParams)
                    currentResidual = cp.copy(newResidual)
                    if currentResidual < bestResidual:
                        bestParams = cp.copy(currentParams)
                else:
                    pass
            T *= 0.98
        print('Annealed after ' + str(step) + ' steps ---------------------------------------------------------------')
        return [np.array(bestParams), bestResidual]

    def hjw(self):
        '''
        Halves jump widths for fitting.
        '''
        self.jumpWidths = (np.array(self.jumpWidths) * .5).tolist()
        
    def PlotGuess(self, components = True):
        '''
        Plots guess fit based on self.paramsGuess
        components = False would stop showing the components on plot.
        '''
        print('Drag around parameters for initial guess, close window once done')
        self.PlotLine(self.paramsGuess, components)

    def PrintParams(self, params, er = []):
        '''
        print either self.paramsGuess or self.fitParams
        '''
        values = []
        errors = []
        for i, v in enumerate(params):
            if i < 6:
                values.append(str(round(v * 1e3, 4)))
                if len(er) != 0:
                    errors.append(str(round(er[i] * 1e3, 4)))
            else:
                values.append(str(round(v, 4)))
                if len(er) != 0:
                    errors.append(str(round(er[i], 4)))
        for i, p in enumerate(self.paramsNames):
            if len(er) != 0:
                print(p + ' = ' + values[i] + ' +- ' + errors[i] + ' ' + self.paramsUnits[i])
            else:
                print(p + ' = ' + values[i] + ' ' + self.paramsUnits[i])

    def Optimise(self, repeats = 20, guessingAfterwards = True):
        '''
        Perform simulated annealing for a set number of seeds (iterations) and pick the best rms value parameters, all others are involved in parameter uncertainties estimation.
        '''
        #Use seed to check consistency (debug purposes)
        np.random.seed(1)
        startingParams = cp.copy(self.paramsGuess)
        startingResidual = self.Residual(startingParams)
        bestParams = []
        bestResiduals = []
        for i in range(repeats):
            fitTemp = self.Anneal()
            bestParams.append(fitTemp[0])
            bestResiduals.append(fitTemp[1])
        fitResidual = np.array(bestResiduals).min()
        bestIndex = np.where(bestResiduals == fitResidual)[0][0]
        self.fitParams = bestParams[bestIndex]
        if fitResidual > startingResidual:
            print('Guess parameters had better residual')
            self.fitParams = startingParams
            fitResidual = startingResidual
        print('Starting residual = ' + str(startingResidual))
        print('Optimised residual = ' + str(fitResidual))
        print('Expected residual = ' + str(1 / self.SNR) + ' (Noise / Normalisation Factor)')
        print('Number of data points = ' + str(self.N))
        print('Number of parameters fitted = ' + str(len(self.boo) - np.sum(self.boo)))
        print('Fitted Params:')
        self.fitParamsEr = np.std(bestParams, axis = 0)
        if self.firstFit == True:
            self.fitParamsErFirstFit = self.fitParamsEr
            self.firstFit = False
        self.PrintParams(self.fitParams, self.fitParamsErFirstFit)
        print('-----------------')
        print('Note these uncertainties are standard deviations of the results from different starting seeds of the FIRST call of Optimise() since NewFit()')
        print('These give an idea of the range of local minima, how well parameters are constrained.')
        print('But will be smaller if the Swing() is small. Or bigger if the initial guess is bad.')
        print('-----------------')
        print('Plotted is the fit of the new parameters, drag around for new guesses if requried.')
        self.paramsGuess = self.fitParams #this is where the guess changes if fitResidual < startingResidual, else they stay the same
        if guessingAfterwards == True:
            self.PlotGuess()


    def PlotLine(self, params, components = False):
        '''
        Interactive model visualisation, best used to find initial guesses!
        Guess parameters are updated whenever a parameter is changed!
        Can change whether a param is held or not in the optimisation, 
        but it doesn't stop you from changing the param on the plot,
        so do not change a parameter if you want it held in the minimisation.
        Reset button doesn't seem to work in iPython... but initial values are indicated by redline.
        '''
        fig, ax = plt.subplots(figsize = (12, 8))
        plt.subplots_adjust(left = 0.2, bottom = 0.4)
        w = self.w
        s = self.FitLine(params)
        (l, ) = plt.plot(w, s, 'r--', lw = 2, label = 'fit')
        if components == True: #strictly after s = self.FitLine(params)
            compNames = []
            for i, c in enumerate(self.components):
                compNames.append('llllllll' + str(int(i + 1))) #add some l's to avoid potential global issues...
                (globals()[compNames[-1]], ) = plt.plot(w, c, 'r:', lw = 1)
        plt.plot(w, self.i, 'ko-', markersize = 5, lw = 2, label = 'data')
        plt.xlabel(r'Wavenumber (cm$^{-1}$)')
        plt.ylabel(r'Intensity (arb.)')
        plt.plot([w[0], w[-1]], [0, 0], color = 'k', lw =.5)
        plt.ylim(-.1, 1.3)
        plt.xticks(np.round(np.linspace(w[0], w[-1], 5), 6))
        plt.legend()
        ax.margins(x = 0)
        axcolor = 'lightgoldenrodyellow'

        rms = self.Residual(params)
        props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)
        rmsText = ax.text(0.03, 0.95, 'rms = ' + str(round(rms, 5)), transform = ax.transAxes, fontsize = 14, verticalalignment = 'top', bbox = props)
        ax.text(0.23, 0.95, 'SNR = ' + str(round(self.SNR, 3)), transform = ax.transAxes, fontsize = 14, verticalalignment = 'top', bbox = props)

        axAu = plt.axes([.2, .27, .6, .02], facecolor = axcolor)
        axAl = plt.axes([.2, .24, .6, .02], facecolor = axcolor)
        axBu = plt.axes([.2, .21, .6, .02], facecolor = axcolor)
        axBl = plt.axes([.2, .18, .6, .02], facecolor = axcolor)
        axGw = plt.axes([.2, .15, .6, .02], facecolor = axcolor)
        axLw = plt.axes([.2, .12, .6, .02], facecolor = axcolor)
        axArea = plt.axes([.2, .09, .6, .02], facecolor = axcolor)
        axCoG = plt.axes([.2, .06, .6, .02], facecolor = axcolor)
        axS = plt.axes([.2, .03, .6, .02], facecolor = axcolor)

        axicut = plt.axes([0.05, .06, .02, .12], facecolor = axcolor)

        #Change slider limits here, units of /cm for A, B, Gw and Lw.
        fmt_str = '%1.4e'
        sAu = Slider(axAu, r'A$_u$ (cm$^{-1}$)', -80e-3, 80e-3, valinit = params[0], valfmt = fmt_str)
        sAl = Slider(axAl, r'A$_l$ (cm$^{-1}$)', -80e-3, 80e-3, valinit = params[1], valfmt = fmt_str)
        sBu = Slider(axBu, r'B$_u$ (cm$^{-1}$)', -40e-3, 40e-3, valinit = params[2], valfmt = fmt_str)
        sBl = Slider(axBl, r'B$_l$ (cm$^{-1}$)', -40e-3, 40e-3, valinit = params[3], valfmt = fmt_str)
        sGw = Slider(axGw, r'G$_w$ (cm$^{-1}$)', 1e-3, 300e-3, valinit = params[4], valfmt = fmt_str)
        sLw = Slider(axLw, r'L$_w$ (cm$^{-1}$)', 1e-4, 50e-3, valinit = params[5], valfmt = fmt_str)
        sArea = Slider(axArea, r'Area (arb.)', 0, 2, valinit = params[6], valfmt = fmt_str)
        sCoG = Slider(axCoG, r'CoG wn (cm$^{-1}$)', params[7] - .1, params[7] + .1, valinit = params[7], valfmt = '%5.4f')
        sS = Slider(axS, r'S', 0, 3, valinit = params[8], valfmt = fmt_str)
        sicut = Slider(axicut, 'icut', 0, int(np.floor(self.i.size / 2)), valinit = self.icut, valstep = 1, valfmt = '%1.f', orientation = 'vertical')

        #Hold/Release Sliders
        axAu = plt.axes([.92, .27, .02, .02], facecolor = axcolor)
        axAl = plt.axes([.92, .24, .02, .02], facecolor = axcolor)
        axBu = plt.axes([.92, .21, .02, .02], facecolor = axcolor)
        axBl = plt.axes([.92, .18, .02, .02], facecolor = axcolor)
        axGw = plt.axes([.92, .15, .02, .02], facecolor = axcolor)
        axLw = plt.axes([.92, .12, .02, .02], facecolor = axcolor)
        axArea = plt.axes([.92, .09, .02, .02], facecolor = axcolor)
        axCoG = plt.axes([.92, .06, .02, .02], facecolor = axcolor)
        axS = plt.axes([.92, .03, .02, .02], facecolor = axcolor)

        fmt_str = '%1.f'
        sAuh = Slider(axAu, 'Hold', 0, 1, valinit = self.boo[0], valstep = 1, valfmt = fmt_str)
        sAlh = Slider(axAl, 'Hold', 0, 1, valinit = self.boo[1], valstep = 1, valfmt = fmt_str)
        sBuh = Slider(axBu, 'Hold', 0, 1, valinit = self.boo[2], valstep = 1, valfmt = fmt_str)
        sBlh = Slider(axBl, 'Hold', 0, 1, valinit = self.boo[3], valstep = 1, valfmt = fmt_str)
        sGwh = Slider(axGw, 'Hold', 0, 1, valinit = self.boo[4], valstep = 1, valfmt = fmt_str)
        sLwh = Slider(axLw, 'Hold', 0, 1, valinit = self.boo[5], valstep = 1, valfmt = fmt_str)
        sAreah = Slider(axArea, 'Hold', 0, 1, valinit = self.boo[6], valstep = 1, valfmt = fmt_str)
        sCoGh = Slider(axCoG, 'Hold', 0, 1, valinit = self.boo[7], valstep = 1, valfmt = fmt_str)
        sSh = Slider(axS, 'Hold', 0, 1, valinit = self.boo[8], valstep = 1, valfmt = fmt_str)

        def update(val):
            Au = sAu.val
            Al = sAl.val
            Bu = sBu.val
            Bl = sBl.val
            Gw = sGw.val
            Lw = sLw.val
            Area = sArea.val
            CoG = sCoG.val
            S = sS.val
            Auh = sAuh.val
            Alh = sAlh.val
            Buh = sBuh.val
            Blh = sBlh.val
            Gwh = sGwh.val
            Lwh = sLwh.val
            Areah = sAreah.val
            CoGh = sCoGh.val
            Sh = sSh.val
            icut = sicut.val
            self.icut = int(icut)
            #Change hold values first, note that rms would change because of degrees of freedom!
            sBoo = cp.copy(self.boo[9:]) # S for all transitions
            self.boo = np.array([Auh, Alh, Buh, Blh, Gwh, Lwh, Areah, CoGh, Sh] + sBoo, dtype = 'int').tolist()
            sGuesses = self.paramsGuess[9:] # S for all transitions
            paramsGuess = np.concatenate(([Au, Al, Bu, Bl, Gw, Lw, Area, CoG, S], sGuesses)).tolist()
            for i, b in enumerate(self.boo): #Now change parameter values
                if b == 0:
                    self.paramsGuess[i] = paramsGuess[i]
            l.set_ydata(self.FitLine(self.paramsGuess))
            if components == True:
                for i, n in enumerate(compNames):
                    globals()[n].set_ydata(self.components[i])
            rmsText.set_text('rms = ' + str(round(self.Residual(self.paramsGuess), 5)))
            fig.canvas.draw_idle()

        sAu.on_changed(update)
        sAl.on_changed(update)
        sBu.on_changed(update)
        sBl.on_changed(update)
        sGw.on_changed(update)
        sLw.on_changed(update)
        sArea.on_changed(update)
        sCoG.on_changed(update)
        sS.on_changed(update)
        sicut.on_changed(update)
        sAuh.on_changed(update)
        sAlh.on_changed(update)
        sBuh.on_changed(update)
        sBlh.on_changed(update)
        sGwh.on_changed(update)
        sLwh.on_changed(update)
        sAreah.on_changed(update)
        sCoGh.on_changed(update)
        sSh.on_changed(update)
        
        #Reset button
        resetax = plt.axes([.05, .24, .06, .04])
        button = Button(resetax, 'Reset', color = axcolor, hovercolor = '0.975')
        def reset(event):
            #Release/hold first then values can be reset
            sAuh.reset()
            sAlh.reset()
            sBuh.reset()
            sBlh.reset()
            sGwh.reset()
            sLwh.reset()
            sAreah.reset()
            sCoGh.reset()
            sSh.reset()
            sAu.reset()
            sAl.reset()
            sBu.reset()
            sBl.reset()
            sGw.reset()
            sLw.reset()
            sArea.reset()
            sCoG.reset()
            sS.reset()
        button.on_clicked(reset)     
        self.title = self.upperLevel + '--->' + self.lowerLevel + ' at ' + str(round(self.paramsGuess[7], 3)) + r'cm$^{-1}$ (' + self.dataFile + ')'
        plt.suptitle(self.title)
        plt.show()
        
        
    def SaveF(self, replace = False):
        '''
        If fitting same line, replace = True will replace previous results, 
        but figure gets replaced regardless.
        Keep figure open when running this if on iPython.
        '''
        self.lineDerivSum = self.DerivSum() #final wiggle index, multiply with SNR for weight.
        plt.savefig('./fits/'+self.upperLevel + '---' + self.lowerLevel + ' (' + self.dataFile + ').png')
        if self.fitLogName != None:
            temp = cp.copy(self.paramsGuess)
            temp[:6] *= 1e3
            temp = temp.tolist()
            temp2 = cp.copy(self.fitParamsErFirstFit)
            temp2[:6] *= 1e3
            temp2 = temp2.tolist()
            self.lineWidth = self.w[-1] - self.w[0]
            row = [[self.dataFile, self.upperLevel, self.lowerLevel] + temp + temp2 + [self.icut, self.SNR, self.lineWidth, self.Residual(self.paramsGuess), self.lineDerivSum, self.w[0], self.w[-1]]]
            row = pd.DataFrame(row, columns = ['Spectrum','Upper Level','Lower Level','Au','Al','Bu','Bl','Gw','Lw',
           'Area','CoGWN','Saturation','AuEr','AlEr','BuEr','BlEr','GwEr','LwEr',
           'AreaEr','CoGWNEr','SaturationEr','icut','SNR','Width','RMS','WiggleIndex','StartWN','EndWN'])
            if replace == False:
                newFitLog = self.fitLog.append(row, ignore_index = 1)
            else:
                index = self.fitLog[(self.fitLog['Upper Level'] == self.upperLevel) & (self.fitLog['Lower Level'] == self.lowerLevel) & (self.fitLog['Spectrum'] == self.dataFile)].index
                self.fitLog.loc[index] = row.values.flatten()
                newFitLog = self.fitLog
            newFitLog.to_excel(self.fitLogName)
            self.fitLog = pd.read_excel(self.fitLogName, index_col = 0)         
    
    def LineFig(self, fitLogIndex, nInterp = 1):
        '''
        Plots line and components with transition diagram for fits in fitLog.xlsx.
        nInterp is the number of points to interpolate between each actual data point (cubic spline).
        '''
        log = self.fitLog.loc[fitLogIndex].to_list()
        if log[0] != self.dataFile:
            print('Loading Spectrum...')
            self.dataFile = log[0]
            self.data = np.loadtxt(self.dataFile)
            print('Done')
        self.paramsGuess = np.array(log[3:12])
        self.paramsGuess[:6] *= 1e-3
        
        print('SNR ', int(log[-6]))
        self.upperJ = int(log[1][-1])
        self.lowerJ = int(log[2][-1])
        self.AllowedTransitions()
        
        self.start_wn = log[-2]
        self.end_wn = log[-1]
        
        #cut paste from WNRange
        line = np.array([d for d in self.data if d[0] >= self.start_wn and d[0] <= self.end_wn])
        self.line = cp.copy(line) #original, non-normalised non-interpolated data
        self.normFactor = line[:, 1].max()
        line[:, 1] /= self.normFactor #normalise line by amplitude 
        #number of points between two neghibouring data points to interpolate
        self.w = np.linspace(line[:, 0][0], line[:, 0][-1], nInterp * (len(line[:, 0])) - nInterp + 1)
        self.spline = interp.CS(line[:, 1], line[:, 0])
 
        #------------------------------------------------------
        #This is the intensity of data interpolated and fitted.
        self.i = self.spline.interp(self.w) #small i attribute is the interpolated instensity
        #------------------------------------------------------
        
        self.icut = log[-7]
        fit = self.FitLine(self.paramsGuess)
        
        Au = log[3]
        Al = log[4]
        
        upperFs = np.unique(self.listF[:, 0])
        lowerFs = np.unique(self.listF[:, 1])

        upperY = []
        for i, E in enumerate(upperFs * Au):
            upperY.append((upperFs * Au)[:i+1].sum())
        
        lowerY = []
        for i, E in enumerate(lowerFs * Al):
            lowerY.append((lowerFs * Al)[:i+1].sum())
        
        #print(upperY, lowerY)
        upperY = np.array(upperY) - np.min(upperY) + np.max(lowerY)
        #print(upperY)
        if self.upperJ != 0 and self.lowerJ != 0:
            maxGap = np.max([np.abs(np.diff(upperY)).max(), np.abs(np.diff(lowerY)).max()])
        elif self.upperJ == 0:
            maxGap = np.abs(np.diff(lowerY)).max()
        else:
            maxGap = np.abs(np.diff(upperY)).max()
        upperY += 1.5 * maxGap
        #print(upperY)
        fig, (ax1, ax2) = plt.subplots(2, num = str(log[1]) + ' to ' + str(log[2]),figsize = (4.5, 6.5))
        plt.subplots_adjust(hspace = 0.4)
        
        ax1.axis('off')
        
        ax1.set_xlim(self.w[0], self.w[-1])
        ax1.locator_params(axis='x', nbins=4)
        ax1.ticklabel_format(useOffset=False)
        wnrange = self.w[-1] - self.w[0]
        #make y range 1 to plot easier
        frac = 0.05
        ax1.text( self.w[0] + frac * wnrange, np.max(upperY) + 0.3 * maxGap, log[1])
        ax1.text( self.w[0] + frac * wnrange, np.min(lowerY) - 0.7 * maxGap, log[2])
        ax1.text( self.w[0] + .6 * wnrange, np.max(upperY) + 0.3 * maxGap, 'A = ' + str(round(Au,1)) + ' mK')
        ax1.text( self.w[0] + .6 * wnrange, np.min(lowerY) - 0.7 * maxGap, 'A = ' + str(round(Al,1)) + ' mK')        
        ax1.hlines(upperY.tolist() + lowerY, xmin = self.w[0] + frac * wnrange, xmax = self.w[-1] - frac * wnrange, color = 'k')
        ax1.vlines([self.w[0] + frac * wnrange, self.w[0] + frac * wnrange], ymin = [np.min(upperY), np.min(lowerY)], ymax = [np.max(upperY), np.max(lowerY)], color = 'k')
        fs = 10
        for i, l in enumerate(self.hfsLines):
            uppery = upperY[np.where(upperFs == self.listF[i][0])[0][0]]
            lowery = lowerY[np.where(lowerFs == self.listF[i][1])[0][0]]
            #ax1.arrow(l, uppery, 0, lowery-uppery, width = 1e-3 * wnrange, head_length = 0.05 * yrange, head_width = 0.02 * wnrange, length_includes_head = True, color = 'k')        
            ax1.annotate('', xy=(l, lowery), xytext=(l, uppery), arrowprops = dict(arrowstyle='->', color = 'r', shrinkA = 0, shrinkB = 0))
            if uppery != np.max(upperY):
                if True == False: #(Au * 3) > Al:
                    ax1.text(self.w[0] + frac * wnrange / 2, uppery, str(self.listF[i][0]), verticalalignment='center', horizontalalignment='right', fontsize = fs)
                else: 
                    if uppery == np.min(upperY):
                        ax1.text(self.w[0] + frac * wnrange / 2, uppery, str(self.listF[i][0]), verticalalignment='center', horizontalalignment='right', fontsize = fs)                    
            else:
                ax1.text(self.w[0] + frac * wnrange / 2, uppery, 'F = ' + str(self.listF[i][0]), verticalalignment='center', horizontalalignment='right', fontsize = fs)
            if lowery != np.min(lowerY):
                if True == False:#(Al * 3) > Au:
                    ax1.text(self.w[0] + frac * wnrange / 2, lowery, str(self.listF[i][1]), verticalalignment='center', horizontalalignment='right', fontsize = fs)
                else:
                    if lowery == np.max(lowerY):
                        ax1.text(self.w[0] + frac * wnrange / 2, lowery, str(self.listF[i][1]), verticalalignment='center', horizontalalignment='right', fontsize = fs)
            else:
                ax1.text(self.w[0] + frac * wnrange / 2, lowery, str(self.listF[i][1]), verticalalignment='center', horizontalalignment='right', fontsize = fs)
        ax2.plot(self.w, self.i, 'ko-', markersize = 4)
        ax2.plot(self.w, fit, 'r--', label = 'Fit')
        ax2.axhline(color = 'k', lw = .5)
        ax2.vlines(self.hfsLines, ymin = np.zeros_like(self.relIntensities), ymax = self.relIntensities * 0.8, lw = 1, color = 'r', label = 'Components')
        ax2.set_xlim(self.w[0], self.w[-1])
        ax2.locator_params(axis='x', nbins=4)
        ax2.ticklabel_format(useOffset=False)
        ax2.set_xlabel(r'Wavenumber (cm$^{-1}$)')
        ax2.set_ylabel('Normalised Intensity')
        #ax2.legend()
        #plt.tight_layout(pad = 3)
        plt.show()
