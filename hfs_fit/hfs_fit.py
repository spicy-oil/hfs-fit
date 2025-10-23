import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Required for interactive plots
import copy as cp
import pandas as pd
import os

from matplotlib.widgets import Slider, Button
from astropy.modeling.models import Voigt1D
from . import rel_int as ri

def K(F, J, I):
    '''
    Change in wavenumber from hfs splitting (of a fine-structure level) is given by 
    '''
    return F * (F + 1) - J * (J + 1) - I * (I + 1)

def dE(A, B, K, F, J, I):
    '''
    B should be zero when I or J is 0 or 1/2 to avoid division by zero
    '''
    if J != .5 and J != 0.:
        return 0.5 * A * K + B * ( (3. / 4 ) * K * (K + 1) - J * (J + 1) * I * (I + 1)) / (2 * I * (2*I - 1) * J * (2*J - 1))
    else:
        return 0.5 * A * K

def add_sat(fit, sat):
    '''
    Add saturation effect to fit
    '''
    return fit * np.exp(- sat * fit)

class Analyser:
    def __init__(self, spec='./data/example_spec.csv', spec_name='example_spec', log_path='./data/log.xlsx', nuclear_spin=3.5):
        '''
        input strings can be directories to the files.

        spec is the string of directory to file of spectral data
        (ASCII CSV, first column is wavenumber, second column is S/N)

        log is the string of directory to .xlsx file of previous fit logs (fit results)
        if file does not exist, new one will be created at specified file directory.
        '''
        self.spec_name = spec_name
        self.log_path = log_path
        self.log_columns = ['spec_name','lev_u','lev_l','A_u','A_l','B_u','B_l','G_w','L_w',
        'area','cog_wn','sat','A_u_unc','A_l_unc','B_u_unc','B_l_unc','G_w_unc','L_w_unc',
        'area_unc','cog_wn_unc','sat_unc','icut','snr','fit_range','rms','wiggle_index','wn_start','wn_end']
        self.load_log()

        self.I = nuclear_spin
        print('Loading spectrum...')
        self.spec_file = spec
        if isinstance(spec, str) == True:
            self.data = np.loadtxt(self.spec_file, delimiter = ',')
        else:
            print('Please use an ASCII file for the spectrum.')
            return
        print('Done')

        self.paramsUnits = ['mK','mK','mK','mK','mK','mK','arb.','/cm','arb.']
        self.paramsNames  = ['A_u', 'A_l', 'B_u', 'B_l', 'G_w', 'L_w', 'Area Parameter', 'CoG Wavenumber', 'Saturation Parameter']

        self.A_range = 80e-3
        self.B_range = 40e-3

    def load_log(self):
        '''
        Load log
        '''
        try:
            self.log = pd.read_excel(self.log_path, index_col = 0)
        except:
            self.log = pd.DataFrame({col: pd.Series(dtype=float) for col in self.log_columns})

    def plot_spec(self, wn=None):
        '''
        Plots the spectrum, specifying a wavenumber will plot a +- 1/cm range around it.
        '''
        plt.close(0)
        plt.figure(0)
        plt.plot(self.data[:, 0], self.data[:, 1], label = 'Spectrum')        
        plt.xlabel(r'Wavenumber (cm$^{-1}$)')
        plt.ylabel('Intensity (arb.)')
        plt.title('Spectrum ' + self.spec_file)
        if wn != None:
            plt.xlim(wn - 1, wn + 1)
        plt.grid()
        plt.show()

    def new_fit(self, J=[], lev=[], line_region=[]):
        '''
        Call this whenever you want to fit a new line.
        Resets everything (J, A, B, Gw, etc.) so make sure you save the previous fit.
        
        J is      [lower_J, upper_J]                      floats
        lev is    [lower_level_label, upper_level_label]  strs
        region is [start_wn, end_wn]                      floats covering the line to be fitted
        '''
        self.lower_J, self.upper_J = J
        self.lower_level_label, self.upper_level_label = lev 
        self.start_wn, self.end_wn = line_region
        self.is_fitted(self.upper_level_label, self.lower_level_label)

        self.calc_component_int()
        self.set_swing()
        self.set_wn_range()
        self.first_fit = True
        self.set_hold()
        self.set_guess_params()
        self.line_deriv_sum = self.deriv_sum()

    def calc_component_int(self):
        '''
        Calculates relative intensities of allowed HFS transitions of the fine structure transition.
        '''
        self.listF = ri.calc_component_int(self.I, self.upper_J, self.lower_J) #list of [upperF, lowerF] transitions
        rel_ints = []
        for f in self.listF:
            rel_ints.append(ri.rel_int(self.I, self.upper_J, self.lower_J, f[0], f[1]))
        self.rel_ints = np.array(rel_ints) #intensity ratios for each transition

    def set_swing(self, swing = [1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-3, 5e-2, .1e-2, .1]):
        '''
        Sets Gaussian standard deviations for the jump distribution for each parameter.
        As, Bs, Gw and Lw are input in /cm.
        Same indices as self.params
        This is reset after every call of new_fit()
        '''
        self.jump_widths = swing

    def set_wn_range(self):
        '''
        set wavenumber range for fitting
        '''
        line = np.array([d for d in self.data if d[0] >= self.start_wn and d[0] <= self.end_wn])
        self.line = cp.copy(line) #original, non-normalised non-interpolated data
        self.snr = int(line[:, 1].max())
        line[:, 1] /= self.snr #normalise line by amplitude 

        self.w = line[:, 0] #wavenumbers to be used in fitting    
        self.i = line[:, 1] #intensities to be used in fitting

        self.N = self.w.size #number of points after interpolation

        temp = np.fft.fft(self.i)
        for i, val in enumerate(temp):
            if np.abs(val) < 0.01 * np.max(np.abs(temp)):
                self.icut = i - 1 # -1 seems to give the best index for apodisation, can change this in plot anyway.
                return

    def set_hold(self, b = [0, 0, 0, 0, 0, 0, 0, 0, 1]):
        '''
        Set which parameter to hold in self.paramGuess, 1 to hold, 0 to not hold.
        same indices as self.params
        Can just turn each on and off in the fitting plot.
        '''
        self.boo = b

    def set_guess_params(self, p = [0, 0, 0, 0, 0.150, 0.005, -1, -1, 0]):
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
        self.params = temp
        #Guess B to be zero, since they tend to be small.
        self.params[2] = 0
        self.params[3] = 0
        print('Current rms = ' + str(self.get_residual(self.params)))
   
    def get_residual(self, params, plot=False):
        '''
        rms, to be minimised, defined as the standard deviation of offsets between each observed intensity and model intensity.
        '''
        model_line = self.calc_model_line(params)
        if plot == True:
            plt.figure()
            plt.plot(self.w, self.i * self.snr, 'ko-', label = 'line')
            plt.plot(self.w, model_line* self.snr, 'r--', label = 'fit')
            plt.plot(self.w, (model_line - self.i) * self.snr, 'r', label = 'residual')
            plt.legend()
            plt.grid()
            plt.ylabel('S/N')
            plt.xlabel(r'Wavenumber cm$^{-1}$')
            ax = plt.gca()
            ax.set_xticks(ax.get_xticks()[::2])  # show every 2nd tick
            plt.show()
        return np.sqrt( ((model_line - self.i)**2).sum() / (len(self.w) - len(self.boo) + np.sum(self.boo)) )

    def calc_model_line(self, params):
        '''
        #Calulates the fitted line at given params and wavenumbers
        '''
        upper_F_wn = np.zeros_like(self.rel_ints)
        for i, F in enumerate(self.listF[:, 0]):
            upper_F_wn[i] += dE(params[0], params[2], K(F, self.upper_J, self.I), F, self.upper_J, self.I)

        lower_F_wn = np.zeros_like(self.rel_ints)
        for i, F in enumerate(self.listF[:, 1]):
            lower_F_wn[i] += dE(params[1], params[3], K(F, self.lower_J, self.I), F, self.lower_J, self.I)

        self.hfs_component_wn = upper_F_wn - lower_F_wn + params[7]
        model_line = np.zeros_like(self.w)
        self.components = []
        for i, v in enumerate(self.rel_ints):
            voigt = Voigt1D(self.hfs_component_wn[i], v, params[5], params[4])
            model_line += voigt(self.w) / (np.pi * params[5] / 2) #normalisation of area
            self.components.append(voigt(self.w) / (np.pi * params[5] / 2))
        model_line = model_line * params[6] / np.sum(self.rel_ints)
        self.components = np.array(self.components) * params[6] / np.sum(self.rel_ints)
        model_line = add_sat(model_line, params[8])
        #self.components = add_sat(self.components, params[8])
        self.model_line_no_apodisation = cp.copy(model_line)
        model_line = self.apodise_line(model_line)
        return model_line
        
    def deriv_sum(self):
        '''
        -Used to find how wiggly a line is within self.set_wn_range(), 
        -Usually the more wiggly a line is the better parameters are constrained.
        -Essentially the sum of the magnitude of differences in intensity between 
        neighbouring points divided by resolution.
        '''
        #Normalise x axis
        w = self.w - self.w[0]
        w /= w[-1]
        i = self.calc_model_line(self.params)
        d = np.gradient(i, w)
        return np.abs(d).sum()

    def is_fitted(self, u, l):
        '''
        -Check if fit already been done
        -Inputs in strings for s (spectrum), u (upper lev), l (lower lev)
        '''
        self.load_log()

        for i, r in self.log.iterrows():
            if r.spec_name == self.spec_name:
                if r.lev_u == u:
                    if r.lev_l == l:
                        print('Fit of line already done')

    def apodise_line(self, fit):
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

    def plot_apo(self):
        '''
        Shows line and guess line in interferogram domain, before and after apodisation of set self.icut
        '''
        #temp = self.calc_model_line(self.params) #generates apodised and non-apodised fits
        I_guess = np.fft.fft(self.model_line_no_apodisation)
        I_data = np.fft.fft(self.i)
        plt.plot(self.w, np.abs(I_data), '-ro', label = 'data')
        plt.plot(self.w, np.abs(I_guess), '-bo', label = 'fit')
        plt.plot(self.w, np.abs(self.I_apo_fit), '-ko', label = 'apodised fit')
        plt.legend()
        plt.grid()
        plt.show()

    def sample_new_params(self, params):
        '''
        Given current params, return new trial params based on jump widths and hold booleans.
        '''
        for i, b in enumerate(self.boo):
            if b == 0: #if False in being kept constant
                newTempParam = np.random.normal(params[i], self.jump_widths[i], 1)[0]
                if i >= 8:
                    params[i] = abs(newTempParam) #saturation parameter cannot be negative...
                elif i == 4:
                    params[i] = abs(newTempParam) #Gw can't be negative
                elif i == 5:
                    params[i] = abs(newTempParam) #Lw can't be negative
                else:
                    params[i] = newTempParam
        return params

    def anneal(self):
        '''
        The simulated annealing algorithim, temperature parameter is multiplied by, say, .98 every iteration
        '''
        step = 1
        current_params = self.params
        current_residual = self.get_residual(params = self.params)
        T = current_residual * .2
        #In case of hopping into a worse local minimum, record what was best!
        best_params = cp.copy(current_params)
        lowest_residual = cp.copy(current_residual)
        while (T - 1e-10) > 0:
            step += 1
            new_params = self.sample_new_params(cp.copy(current_params))
            new_residual = self.get_residual(params = new_params)
            residual_change = new_residual - current_residual
            if residual_change <= 0:
                print('Residual decreased! Current residual = ' + str(new_residual) + ' step = ' + str(step))
                current_params = cp.copy(new_params)
                current_residual = cp.copy(new_residual)
                if current_residual < lowest_residual:
                    best_params = cp.copy(current_params)
                    lowest_residual = cp.copy(current_residual)
            else:
                boltz_prob = np.exp(-residual_change/T)
                if np.random.uniform() <= boltz_prob:
                    print('Randomly Excited!')
                    current_params = cp.copy(new_params)
                    current_residual = cp.copy(new_residual)
                    if current_residual < lowest_residual:
                        best_params = cp.copy(current_params)
                else:
                    pass
            T *= 0.98
        print('Annealed after ' + str(step) + ' steps ---------------------------------------------------------------')
        return [np.array(best_params), lowest_residual]

    def hjw(self):
        '''
        Halves jump widths for fitting.
        '''
        self.jump_widths = (np.array(self.jump_widths) * .5).tolist()
        
    def print_params(self, params, er=[]):
        '''
        print either self.params or self.fit_params
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

    def optimise(self, repeats=20, guessingAfterwards=True):
        '''
        Perform simulated annealing for a set number of seeds (iterations) and pick the best rms value parameters, all others are involved in parameter uncertainties estimation.
        '''
        #Use seed to check consistency (debug purposes)
        np.random.seed(1)
        starting_params = cp.copy(self.params)
        starting_residual = self.get_residual(starting_params)
        best_params = []
        lowest_residuals = []
        for i in range(repeats):
            temp = self.anneal()
            best_params.append(temp[0])
            lowest_residuals.append(temp[1])
        fit_residual = np.array(lowest_residuals).min()
        best_index = np.where(lowest_residuals == fit_residual)[0][0]
        self.fit_params = best_params[best_index]
        if fit_residual > starting_residual:
            print('Guess parameters had better residual')
            self.fit_params = starting_params
            fit_residual = starting_residual
        print('Starting residual = ' + str(starting_residual))
        print('Optimised residual = ' + str(fit_residual))
        print('Expected residual = ' + str(1 / self.snr) + ' (Noise / Normalisation Factor)')
        print('Number of data points = ' + str(self.N))
        print('Number of parameters fitted = ' + str(len(self.boo) - np.sum(self.boo)))
        print('Fitted Params:')
        self.fit_params_er = np.std(best_params, axis=0)
        
        # First fit has largest uncertainties from initial guess, later fits are much more local
        if self.first_fit == True:
            self.fit_params_er_first_fit = self.fit_params_er
            self.first_fit = False
        self.print_params(self.fit_params, self.fit_params_er_first_fit)
        print('-----------------')
        print('Note these uncertainties are standard deviations of the results from different starting seeds of the FIRST call of optimise() since new_fit()')
        print('These give an idea of the range of local minima, how well parameters are constrained.')
        print('But will be smaller if the swing() is small. Or bigger if the initial guess is bad.')
        print('-----------------')
        print('Plotted is the fit of the new parameters, drag around for new guesses if requried.')
        self.params = self.fit_params #this is where the guess changes if fit_residual < starting_residual, else they stay the same
        if guessingAfterwards == True:
            self.plot_fit()

    def plot_fit(self, A_range=80, B_range=40, components=True, save=False, output_dir=None):
        '''
        Interactive model visualisation, best used to find initial guesses!
        Guess parameters are updated whenever a parameter is changed!
        Can change whether a param is held or not in the optimisation, 
        but it doesn't stop you from changing the param on the plot,
        so do not change a parameter if you want it held in the minimisation.
        Reset button doesn't seem to work in iPython... but initial values are indicated by redline.
        '''
        params = self.params

        fig, ax = plt.subplots(figsize = (12, 8))
        plt.subplots_adjust(left = 0.2, bottom = 0.4)
        w = self.w
        s = self.calc_model_line(params)
        (l, ) = plt.plot(w, s, 'r--', lw = 2, label = 'fit')
        if components == True: #strictly after s = self.calc_model_line(params)
            component_lines = {}
            for i, c in enumerate(self.components):
                line, = plt.plot(w, c, 'r:', lw=1)
                component_lines[f'component_{i}'] = line
        plt.plot(w, self.i, 'ko-', markersize = 5, lw = 2, label = 'data')
        plt.xlabel(r'Wavenumber (cm$^{-1}$)')
        plt.ylabel(r'Intensity (arb.)')
        plt.plot([w[0], w[-1]], [0, 0], color = 'k', lw =.5)
        plt.ylim(-.1, 1.3)
        plt.xticks(np.round(np.linspace(w[0], w[-1], 5), 6))
        plt.legend()
        ax.margins(x = 0)
        axcolor = 'lightgoldenrodyellow'

        rms = self.get_residual(params)
        props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)
        rmsText = ax.text(0.03, 0.95, 'rms = ' + str(round(rms, 5)), transform = ax.transAxes, fontsize = 14, verticalalignment = 'top', bbox = props)
        ax.text(0.23, 0.95, 'SNR = ' + str(round(self.snr, 3)), transform = ax.transAxes, fontsize = 14, verticalalignment = 'top', bbox = props)

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
        constants_fmt = "%.2f"
        self.A_range = A_range * 1e-3
        self.B_range = B_range * 1e-3
        sAu = Slider(axAu, r'A$_u$ (mK)', -self.A_range, self.A_range, valinit = params[0], valfmt = constants_fmt)
        sAl = Slider(axAl, r'A$_l$ (mK)', -self.A_range, self.A_range, valinit = params[1], valfmt = constants_fmt)
        sBu = Slider(axBu, r'B$_u$ (mK)', -self.B_range, self.B_range, valinit = params[2], valfmt = constants_fmt)
        sBl = Slider(axBl, r'B$_l$ (mK)', -self.B_range, self.B_range, valinit = params[3], valfmt = constants_fmt)
        sGw = Slider(axGw, r'G$_w$ (mK)', 1e-3, 300e-3, valinit = params[4], valfmt = constants_fmt)
        sLw = Slider(axLw, r'L$_w$ (mK)', 1e-4, 50e-3, valinit = params[5], valfmt = constants_fmt)
        sArea = Slider(axArea, r'Area (arb.)', 0, 2, valinit = params[6], valfmt = '%.2f')
        sCoG = Slider(axCoG, r'CoG wn (cm$^{-1}$)', params[7] - .1, params[7] + .1, valinit = params[7], valfmt = '%5.4f')
        sS = Slider(axS, r'S', 0, 3, valinit = params[8], valfmt = '%.2f')
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
            sAu.valtext.set_text(f"{sAu.val * 1e3:.2f}")  # scale to mK
            Al = sAl.val
            sAl.valtext.set_text(f"{sAl.val * 1e3:.2f}")  # scale to mK
            Bu = sBu.val
            sBu.valtext.set_text(f"{sBu.val * 1e3:.2f}")  # scale to mK
            Bl = sBl.val
            sBl.valtext.set_text(f"{sBl.val * 1e3:.2f}")  # scale to mK
            Gw = sGw.val
            sGw.valtext.set_text(f"{sGw.val * 1e3:1.1f}")  # scale to mK
            Lw = sLw.val
            sLw.valtext.set_text(f"{sLw.val * 1e3:1.1f}")  # scale to mK
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
            sGuesses = self.params[9:] # S for all transitions
            params = np.concatenate(([Au, Al, Bu, Bl, Gw, Lw, Area, CoG, S], sGuesses)).tolist()
            for i, b in enumerate(self.boo): #Now change parameter values
                if b == 0:
                    self.params[i] = params[i]
            l.set_ydata(self.calc_model_line(self.params))
            if components == True:
                for i, (key, line) in enumerate(component_lines.items()):
                    line.set_ydata(self.components[i])
            rmsText.set_text('rms = ' + str(round(self.get_residual(self.params), 5)))
            fig.canvas.draw_idle()
        
        # ensure correct initial display
        update(None)
        
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
        self.title = self.lower_level_label + r' $-$ ' + self.upper_level_label + ' at ' + str(round(self.params[7], 3)) + r'cm$^{-1}$ (' + self.spec_name + ')'
        plt.suptitle(self.title)
        if save == True:
            plt.savefig(output_dir)
            plt.close()
        else:
            plt.show()
        
        
    def save_fit(self, replace=False):
        '''
        Writes to log.xlsx and saves figure
        If fitting same line, replace=True will replace previous results, 
        but figure gets replaced regardless.
        '''
        self.load_log()
            
        self.line_deriv_sum = self.deriv_sum() #final wiggle index, multiply with SNR for weight.
        fig_name = self.lower_level_label + r' $-$ ' + self.upper_level_label + ' (' + self.spec_name + ').png'
        
        directory = os.path.dirname(self.log_path)
        self.plot_fit(components=True, save=True, output_dir=os.path.join(directory, fig_name))

        temp = cp.copy(self.params)
        temp[:6] *= 1e3
        temp = temp.tolist()
        temp2 = cp.copy(self.fit_params_er_first_fit)
        temp2[:6] *= 1e3
        temp2 = temp2.tolist()
        self.fit_range = self.w[-1] - self.w[0]
        row = [[self.spec_name, self.upper_level_label, self.lower_level_label] + 
                temp + temp2 + [self.icut, self.snr, self.fit_range, self.get_residual(self.params), 
                                self.line_deriv_sum, self.w[0], self.w[-1]]]
        row = pd.DataFrame(row, columns=self.log_columns)
        if replace == False:
            new_log = pd.concat([self.log, row], ignore_index=True)
        else:  # Replacing fit results
            index = self.log[(self.log['lev_u'] == self.upper_level_label) & 
                                (self.log['lev_l'] == self.lower_level_label) & 
                                (self.log['spec_name'] == self.spec_name)].index
            self.log.loc[index] = row.values.flatten()
            new_log = self.log
        new_log.to_excel(self.log_path)
        self.log = pd.read_excel(self.log_path, index_col=0)         
    
    def transitions_fig(self, log_index):
        '''
        Plots line and components with transition diagram for fits in log.xlsx.
        nInterp is the number of points to interpolate between each actual data point (cubic spline).
        '''
        self.load_log()

        log = self.log.loc[log_index].to_list()
        self.params = np.array(log[3:12])
        self.params[:6] *= 1e-3
        
        print('SNR ', int(log[-6]))
        self.upper_J = int(log[1][-1])
        self.lower_J = int(log[2][-1])
        self.calc_component_int()
        
        self.start_wn = log[-2]
        self.end_wn = log[-1]
        
        #cut paste from set_wn_range
        line = np.array([d for d in self.data if d[0] >= self.start_wn and d[0] <= self.end_wn])
        self.line = cp.copy(line) #original, non-normalised non-interpolated data
        self.normFactor = line[:, 1].max()
        line[:, 1] /= self.normFactor #normalise line by amplitude 
        #number of points between two neghibouring data points to interpolate
        self.w = line[:, 0]
        self.i = line[:, 1]
        
        self.icut = log[-7]
        fit = self.calc_model_line(self.params)
        
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
        if self.upper_J != 0 and self.lower_J != 0:
            maxGap = np.max([np.abs(np.diff(upperY)).max(), np.abs(np.diff(lowerY)).max()])
        elif self.upper_J == 0:
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
        for i, l in enumerate(self.hfs_component_wn):
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
        ax2.vlines(self.hfs_component_wn, ymin = np.zeros_like(self.rel_ints), ymax = self.rel_ints * 0.8, lw = 1, color = 'r', label = 'Components')
        ax2.set_xlim(self.w[0], self.w[-1])
        ax2.locator_params(axis='x', nbins=4)
        ax2.ticklabel_format(useOffset=False)
        ax2.set_xlabel(r'Wavenumber (cm$^{-1}$)')
        ax2.set_ylabel('Normalised Intensity')
        #ax2.legend()
        #plt.tight_layout(pad = 3)
        plt.show()
