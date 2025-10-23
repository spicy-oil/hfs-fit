# HFS-FIT
**An interactive python fitting program for atomic emission lines with hyperfine structure (HFS), the associated article is [here](https://doi.org/10.3847/1538-4365/abbdf8).**

**It has sliders for parameters to change the fit visually, useful when HFS constants are unknown and we need initial guesses for optimisation (annealing included in the package).**


<img src="/data/z5S2---a5P2 (example_spec).png">

# Quickstart

Developed in python 3.6, updated for python 3.11.11. To install with dependencies in a chosen environment run 
```
pip install -e .
```
use optional `-e` if want source code modifications to realise in new imports.

Run step-by-step content in `example.py` in an interactive python environment (need access to matplotlib interactive windows).

# Files and Explanations

1) `example.py` - basic usage.

2) `log.xlsx` - parameters saved here when desired.

3) `hfs_fit.py` - Main script that makes use of others, contains class for spectrum, contains fitting, plotting algorithms.

4) `rel_int.py` - routine to calculate relative intensities of HFS components, used by hfs_fit.py.

5) `example_spec.csv` - example spectrum input covering a small portion of an UV Co II spectrum with 4 Co II lines.

6) `hfs_old` - old repository of previous release.

# Useful Functions and Notes

- The `Analyser.transitions_fig()` function plots transition diagram with HF components (for recorded fits in a `log.xlsx`). The spacing between texts may not be perfect, most of the time the level label will touch a level line, can change this by changing the location of the texts in the code. Example:
<img src="/data/z5S2---a5P2 (example_spec) diagram.png">

- The `Analyser.plot_spec()` function will plot the whole spectrum, input a wavenumber in the bracket and it will plot around that wavenumber. Example:
<img src="/data/spec_plot.png">

- The `Analyser.get_residual()` function can plot current fit and residual: `Analyser.get_residual(Analyser.params, plot=True)`. Example (please excuse the x-axis):
<img src="/data/residual_plot.png">

- Use `Analyser.hjw()` to half all jumpwidths before `Analyser.optimise()`, this is convenient when performing the final optimisation of parameters, or if the initial guess is very good.

- Can always re-open the sliders plot with `Analyser.plot_fit()`, and adjust then `Analyser.optimise()` as needed.

- HFS components are plotted by default, can turn this off using `Analyser.plot_fit(components=False)`.

- If the finite optical path instrumental profile (Fourier transform spectroscopy only) is negligible, set `icut` at the maximum value.
