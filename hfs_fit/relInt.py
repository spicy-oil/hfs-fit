import numpy as np

#Need to calculate relative intensities of the transitions from the two given levels!

def P(A, B, C):
    return (A + B) * (A + B + 1) - C * (C + 1)

def Q(A, B, C):
    return C * (C + 1) - (A - B) * (A - B + 1)

def R(A, B, C):
    return A * (A + 1) + B * (B + 1) - C * (C + 1)

def Temp(I, J1Value, J2Value, F1, F2):
    '''
    A function used twice in RelIntensity
    '''
    IF1 = 2 * F1 + 1
    IF2 = 2 * F2 + 1
    if IF1 == IF2:
        if F1 == 0:
            C1 = 0 #This is to avoid nan intensity from F1 = 0 and F2 = 0
        else:
            C1 = (2 * F1 + 1) / (F1 * (F1 + 1))
    elif abs(IF2-IF1) != 2: #pretty sure this bit is the transition rule where change in F is no greater than 1
        return 0
    if J1Value == J2Value: #only in this case can F1 = F2 = 0, if not, C1 is not used so we are fine with setting C1 to 0 when F1 = F2 = 0
        if IF1 == IF2:
            output = C1 * R(F1, J1Value, I) ** 2
        elif IF1 < IF2:
            output = P(F2, J1Value, I) * Q(F1, J1Value, I) / F2
        else:
            output = P(F1, J1Value, I) * Q(F2, J1Value, I) / F1
    elif J1Value > J2Value:
        if IF1 == IF2:
            output = C1 * P(F1, J1Value, I) * Q(F1, J1Value, I)
        elif IF1 < IF2:
            output = Q(F2, J1Value, I) * Q(F1, J1Value, I) / F2
        else:
            output = P(F1, J1Value, I) * P(F2, J1Value, I) / F1
    else:
        if IF1 == IF2:
            F4 = F1 + 1
            F3 = F1 - 1
            output = C1 * P(F4, J1Value, I) * Q(F3, J1Value, I)
        elif IF1 < IF2:
            F4 = F2 + 1
            output = P(F2, J1Value, I) * P(F4, J1Value, I) / F2
        else:
            F4 = F2 - 1
            output = Q(F2, J1Value, I) * Q(F4, J1Value, I) / F1
    return output

def RelIntensity(I, J1Value, J2Value, F1Value, F2Value):
    '''
    Calculates relative intensity of a transition of given Js and Fs in the atom with spin I
    Doesn't matter 1 or 2 is upper, just need to be consistent across Ji and Fi
    Taken from Xgremlin source code
    '''
    F1 = J1Value + I #these are max intensity Fs
    F2 = J2Value + I
    FMAX = Temp(I, J1Value, J2Value, F1, F2)
    if FMAX == 0:
        return 0
    output = Temp(I, J1Value, J2Value, F1Value, F2Value)
    return output / FMAX

def AllowedTransitions(I, upperJ, lowerJ):
    upperFs = np.arange(abs(upperJ - I), upperJ + I + 1, 1)
    lowerFs = np.arange(abs(lowerJ - I), lowerJ + I + 1, 1)
    transitions = []
    for fu in upperFs:
        for fl in lowerFs:
            if abs(fu - fl) < 2:
                transitions.append([fu, fl])
    return np.array(transitions)

