'''define all the function used in ProjB
'''
from __future__ import division
import numpy as np
import math #YOU NEED CHANGE ALL MATH MODULE TO EITHER NUMPY or SCIPY
import scipy as sp
from scipy.special import erfc


def readdata(filename):
    t = []
    sigma = []
    with open(filename, 'rb') as f:
        for line in f:
            t.append(float(line.split()[0]))
            sigma.append(float(line.split()[1]))
    return np.array(t), np.array(sigma)

t_m, sigma_m = readdata('lifetime.txt')


def fmsignal(tau, t, sigma):
    return (1/(2*tau)) \
        * sp.exp((sigma**2/(2*tau**2))-(t/tau)) \
        * erfc((1/math.sqrt(2))*((sigma/tau) - (t/sigma)))


def NLL(tau, time_list=t_m, sigma_list=sigma_m):
    fmt = fmsignal(tau, time_list, sigma_list)
    return np.sum(-np.log(fmt))


def parabolicMin(func,xlist):
    '''Example:
    >>> parabolicMin(math.cosh,[-1.2,1.2,1])
    >>> parabolicMin(NLL,[0.2,0.4,0.5])
    '''

    while max(xlist)-min(xlist) > 1e-3:
        ylist = [func(xlist[0]), func(xlist[1]), func(xlist[2])]
        upper = (xlist[2]**2 - xlist[1]**2)*ylist[0] + (xlist[0]**2 - xlist[2]**2)*ylist[1] + (xlist[1]**2 - xlist[0]**2)*ylist[2]
        lower = (xlist[2] - xlist[1])*ylist[0] + (xlist[0] - xlist[2])*ylist[1] + (xlist[1] - xlist[0])*ylist[2]
        x3 = 0.5 * (upper/lower)
        y3 = func(x3)
        if y3 < max(ylist):
            i = ylist.index(max(ylist))
            ylist[i] = y3
            xlist[i] = x3
    return xlist[ylist.index(min(ylist))], min(ylist), xlist,ylist


def fmbackground(t, sigma):
    return sp.exp(-0.5*(t**2/sigma**2))/(sigma*math.sqrt(2*math.pi))


def ft(tau, signalfrac, t, sigma):
    return signalfrac*fmsignal(tau, t, sigma) + (1-signalfrac)*fmbackground(t, sigma)


def curvature(function_array):
    f1 = np.gradient(function_array)
    f2 = np.gradient(f1)
    return np.abs(f2/((1+f1**2)**1.5))


def derivative2(function_array):
    return np.gradient(np.gradient(function_array))


def stderror(func):
    pass

def NLL2D(tau,signalfrac):
    pass