#%%
from hfs_fit import Analyser

#%% Step 1: Create an instance of the Analyser class for the spectrum (ensure in SNR scale)
a = Analyser(spec='./data/example_spec.csv', spec_name='example_spec', 
             log_path='./data/log.xlsx', nuclear_spin=3.5)
# log will be created if not exists

#%% Step 2: Set up a new fit with specified parameters [J_u, J_l], [upper level label, lower level label], and line region
a.new_fit(J=[2, 2], lev=['z5S2', 'a5P2'], line_region=[37978, 37980])
# all need to be specified for a new fit

#%% Step 3: Plot initial guess and adjust parameters interactively
a.plot_fit(A_range=80, B_range=40)  # specify ranges for A and B constants in mK, these are default values
# adjust parameters with sliders if needed, then close the plot window to proceed

#%% Step 4: Optimize parameters multiple times with different seeds
a.optimise(5)  # multiple seeds to avoid local minima and get variance estimates, I typically use 25

# plot with sliders will show again, just close it if looks fine, or adjust and fit again...

#%% Step 5: Save the fit results and the plot
a.save_fit(replace=False)  # not replacing exisiting fit results for the same line
# output file will be in the same dir as log_path

#%% Load fit and plot transition diagram
a.transitions_fig(log_index=0)  # specify which fit to plot based on log index
# make sure the correct spectrum is loaded if multiple spectra are used
# can save the figure from the pop-up window or from more programming (not included in the package)

# %% Continue
# a.new_fit() # Uncomment to start a new fit on the same spectrum if needed

