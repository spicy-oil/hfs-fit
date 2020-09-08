import hfs_fit as h

c = h.hfs(dataASCII = 'spectrum.txt', fitLog = 'fitLog.xlsx', nuclearSpin = 3.5) #Instance for fitting
c.NewFit() #Start a new fit, this resets line info, guess parameters, jumpwidths and fit results.

'''
----Step 1----:
    
Input these after the script is run.
These are the information regarding the line.

upperJ: 2
lowerJ: 2

upperLevel: z5S2
lowerLevel: a5P2

startingWN: 37978
endWN: 37980

noiseStart: 37945
noiseEnd: 37975

Can change SNR by changing the attribute c.SNR to the SNR float of choice after this.

----Step 2----:
    
Starting guessing initial values typing in console:

c.PlotGuess()

Drag A_u to -8e-3 and A_l = 51e-3, icut to max for best guess for this line.

Close once done guessing.

----Step 3----:
    
Optimise parameters 5 times, each time with different seed by inputing

c.Optimise(5)

----Step 4----:
If wish to save this fit into fitLog.xlsx and as an image under /fits use (keep the plot open):

c.SaveF()

----Step 5----:
For another fit of the loaded spectrum:
    
c.NewFit()

If fitting another spectrum, remake the instance to avoid clashes in SaveF() 
as there is only one file of fitLog.xlsx:
    
c = h.hfs(spectrumFile)
'''