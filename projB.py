"""Name: projB
Purpose: contains functions necessary for 1D and 2D function minimisation to estimate the life time of meson D^0
Author: Yichen Liu
Created: 23/11/2016

Important Notes
---------------
- This module requires numpy and scipy packages to run, and the functions in this module performs the best
  when they are used on numpy.array types of data.
- For practicality, this module automatically intake meson D^0 measurements data from file 'lifetime.txt',
  ensure this file is in the same directory as this module.

Functions
----------
- read_data: read in .txt file of data in two columns.
- f_signal:
- f_total:
- NLL:
- NLL2D:
- min_parabolic:
- min_gradient_descent:
- min_newton:
- min_quasi_newton:
- std_error:

ASK JOJO IF SHOULD MERGE TWO NLL FUNCTIONS!!!!!!
"""
from __future__ import division

__all__ = ['read_data', 'f_signal', 'f_total', 'NLL', 'NLL2D',
           'min_parabolic', 'min_gradient_descent', 'min_quasi_newton',
           'min_newton', 'std_error', 't_m', 'sigma_m']

import numpy as np
import scipy as sp
from scipy.special import erfc
import numpy.linalg as la
import copy


def read_data(filename):
    """Read in .txt file of float type data in two columns. returns each column as a numpy.array.

    Parameters
    ----------
    filename : string
        Name of the file with extension as a string. e.g. 'lifetime.txt'

    Returns
    -------
    t : 1D numpy.array
        Data in the first column.
    sigma : 1D numpy.array
        Data in the second column.
    """
    t = []
    sigma = []
    with open(filename, 'rb') as f:
        for line in f:
            t.append(float(line.split()[0]))
            sigma.append(float(line.split()[1]))
    return np.array(t), np.array(sigma)


t_m, sigma_m = read_data('lifetime.txt')  # permit direct access to lifetime.txt data


def f_signal(tau, t, sigma):
    """Probability density function of the decay signal. Obtained by convolution of
    exponential distribution of the decay (time t, average time tau)
    and Gaussian function (due to detection) of width sigma.

    Parameters
    ----------
    tau : float or 1D array
        Average decay time.
    t : float or 1D array
        Measured decay time.
    sigma : float or 1D array
        Measurement error.

    Returns
    -------
    f_s : float or 1D array
        Probability density associated with the input parameters.
    """
    f_s = (1/(2*tau)) \
        * sp.exp((sigma**2/(2*tau**2))-(t/tau)) \
        * erfc((1/np.sqrt(2))*((sigma/tau) - (t/sigma)))
    return f_s


def _noise(t, sigma):
    """Noise is the convolution of a delta function with a Gaussian (due to detection), it's just the Gaussian itself.

    Parameters
    ----------
    t : float or 1D array
        Measured decay time.
    sigma : float or 1D array
        Measurement error.

    Returns
    -------
    Gaussian : float or 1D array
        Probability density associated with the input parameters.
    """
    return sp.exp(-0.5*(t**2/sigma**2))/(sigma*np.sqrt(2*np.pi))


def f_total(tau, signal_fraction, t, sigma):
    """Probability density function of the decay signal including background noise.
    Obtained by summing a fraction of PDF of pure signal and a (1-fraction) background noise.

    Parameters
    ----------
    tau : float or 1D array
        Average decay time.
    signal_fraction : float or 1D array
        Proportion of signal in detection. only defined between 0 and 1.
    t : float or 1D array
        Measured decay time.
    sigma : float or 1D array
        Measurement error.

    Returns
    -------
    f_t : float or 1D array
        Probability density associated with the input parameters.

    Raises
    ------
    ValueError
        If signal_fraction exceeds 1.
    """
    if signal_fraction > 1. or signal_fraction < 0.:
        raise ValueError("The signal fraction is only defined between 0 and 1.")
    f_t = signal_fraction * f_signal(tau, t, sigma) + (1 - signal_fraction) * _noise(t, sigma)
    return f_t


def NLL(tau, time_list=t_m, sigma_list=sigma_m):
    """Negative log likelihood of f_signal function (ref. help(f_signal) ) to estimate tau.

    Parameters
    ----------
    tau : float
        Average decay time.
    time_list : 1D array, optional
        Measured decay time. default use data obtained from lifetime.txt.
    sigma_list : 1D array, optional
        Measurement error. default use data obtained from lifetime.txt.

    Returns
    -------
    L : float
        Likelihood of tau.
    """
    L = np.sum(-np.log(f_signal(tau, time_list, sigma_list)))
    if np.isnan(L):
        raise ValueError('Result is not a number.')
    return L


def NLL2D(tau, signal_fraction, time_list=t_m, sigma_list=sigma_m):
    """Negative log likelihood of f_total function (ref. help(f_total) ) to estimate tau.

    Parameters
    ----------
    tau : float
        Average decay time.
    signal_fraction : float
        Proportion of signal in detection. only defined between 0 and 1.
    time_list : 1D array, optional
        Measured decay time. default use data obtained from lifetime.txt.
    sigma_list : 1D array, optional
        Measurement error. default use data obtained from lifetime.txt.

    Returns
    -------
    L : float
        Likelihood of tau and signal_fraction.
    """
    L = np.sum(-np.log(f_total(tau, signal_fraction, time_list, sigma_list)))
    if np.isnan(L):
        raise ValueError('Result is not a number.')
    return L


def min_parabolic(func, xlist, tol=1e-5):
    """Example:
    >>> min_parabolic(math.cosh,[-1.2,1.2,1])
    (-0.0,
 1.0,
 [-0.0, 1.359745106347053e-06, 7.118859805199238e-11],
 [1.0, 1.0000000000009244, 1.0])

    >>> min_parabolic(NLL,[0.2,0.4,0.5])
    (0.40454571849127174,
 6220.4468927881981,
 [0.40454571849127174, 0.4045458573036621, 0.40454654592198747],
 [6220.4468927881981, 6220.446892788651, 6220.4468928037604])
    """

    while max(xlist)-min(xlist) >= tol:
        ylist = [func(xlist[0]), func(xlist[1]), func(xlist[2])]
        upper = (xlist[2]**2 - xlist[1]**2)*ylist[0] + (xlist[0]**2 - xlist[2]**2)*ylist[1] + (xlist[1]**2 - xlist[0]**2)*ylist[2]
        lower = (xlist[2] - xlist[1])*ylist[0] + (xlist[0] - xlist[2])*ylist[1] + (xlist[1] - xlist[0])*ylist[2]
        x3 = 0.5 * (upper/lower)
        y3 = func(x3)
        if y3 < max(ylist):
            i = ylist.index(max(ylist))
            ylist[i] = y3
            xlist[i] = x3
    return xlist[ylist.index(min(ylist))], min(ylist), xlist, ylist


def _partial2D(func, x, delta=1e-5):
    """Partial derivative for a function of two parameters i.e. f(x_0, x_1)
    using central difference approximation.

    Parameters
    ----------
    func : callable
        Objective function.
    x : 1D array of two elements
        Point at which the gradient is returned. e.g. numpy.array([x_0, x_1])
    delta : float, optional
        Step size in the central difference approximation, in both x_0 and x_1 directions.

    Returns
    -------
    gradient : 1D array of two elements
        Gradient at x of func.
    """
    diff = 2*delta
    gradient = np.array([((func(x[0]+delta, x[1]) - func(x[0]-delta, x[1])) / diff),
                        ((func(x[0], x[1]+delta) - func(x[0], x[1]-delta)) / diff)])
    return gradient


def _hessian2D(func, x, delta=1e-4):
    """2x2 Hessian for a function of two parameters i.e. f(x_0, x_1)
    using central difference approximation. Thus 2n+4n^2/2 additional functions calls are needed.

    Parameters
    ----------
    func : callable
        Objective function.
    x : 1D array of two elements
        Point at which the gradient is returned. e.g. numpy.array([x_0, x_1])
    delta : float, optional
        Step size in the central difference approximation, in both x_0 and x_1 directions.

    Returns
    -------
    H : 2x2 numpy array
        Hessian at x of func.

    Notes
    -----
    Assumed df/dxdy is same as df/dydx, which is true for well behaved function, i.e. almost all functions.
    """
    d2f_dx2 = (-func(x[0] + 2 * delta, x[1]) + 16 * func(x[0] + delta, x[1]) - 30 * func(x[0], x[1])
               + 16 * func(x[0] - delta, x[1]) - func(x[0] - 2 * delta, x[1])) / (12 * delta ** 2)
    d2f_dy2 = (-func(x[0], x[1] + 2 * delta) + 16 * func(x[0], x[1] + delta) - 30 * func(x[0], x[1])
               + 16 * func(x[0], x[1] - delta) - func(x[0], x[1] - 2 * delta)) / (12 * delta ** 2)
    d2f_dxdy = (func(x[0] + delta, x[1] + delta) - func(x[0] + delta, x[1] - delta)
                - func(x[0] - delta, x[1] + delta) + func(x[0] - delta, x[1] - delta)) / (4 * delta ** 2)
    #same as d2f_dydx
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])


def min_gradient_descent(func, start, alpha=1e-6, tol=1e-10, maxiter=1e4):
    x = copy.copy(start)
    niter = 0
    improved = True
    while improved and niter < maxiter:
        niter += 1
        print _partial2D(func, x)
        step = alpha * _partial2D(func, x)
        x -= step
        print x, step, niter
        if la.norm(step) < tol:
            improved = False
    return x, niter


def min_newton(func, start, tol=1e-10, maxiter=100, bound_mode=False):
    """input dtype=float"""
    x = copy.copy(start)
    niter = 0
    improved = True
    while improved and niter < maxiter:
        niter += 1
        H = _hessian2D(func, x)
        step = np.dot(-la.inv(H), _partial2D(func, x))
        if bound_mode:
            up_diff = 1. - x[1]
            while step[1] > up_diff:
                step[1] *= 0.9
        x += step
        print x, step
        print H
        if la.norm(step) < tol:
            improved = False
    return x, niter


def min_quasi_newton(func, start, alpha=1e-6, tol=1e-8, maxiter=1e4):
    x = copy.copy(start)
    niter = 0
    improved = True
    G = np.identity(2)
    while improved and niter < maxiter:
        niter += 1
        x_prime = _partial2D(func, x)
        step = - alpha * np.dot(x_prime, G)
        x += step
        if la.norm(step) < tol:
            improved = False
        gamma = _partial2D(func, x) - x_prime
        G += np.outer(step, step) / np.dot(gamma, step) \
            - np.dot(G, np.dot(np.outer(step, step), G)) / np.dot(gamma, np.dot(G, gamma))
    return x, niter


def _curv(x, y):
    """Curvature of Lagrange second polynomial, i.e. the second order derivative. Calculated analytically.

    Parameters
    ----------
    x : list of three elements
        Three points that was used to estimate the polynomial. e.g. [x_0, x_1, x_2]
    y : list of three elements
        Correspondent f(x).

    Returns
    -------
    c : float
        Curvature of the parabola.
    """
    d = (x[1]-x[0])*(x[2]-x[0])*(x[2]-x[1])
    return (x[2]-x[1])*y[0]/d + (x[0]-x[2])*y[1]/d + (x[1]-x[0])*y[2]/d


def std_error(func, x, method='curvature', decimals=4): #x being a list
    x_min, y_min, xlist, ylist = min_parabolic(func, x)
    if method == 'curvature':
        a = _curv(xlist, ylist)
        return np.around(np.sqrt(1/(2*a)), decimals=decimals)
    elif method == 'scan':
        y_min_up = copy.copy(y_min)
        y_min_down = copy.copy(y_min)
        x_up = copy.copy(x_min)
        x_down = copy.copy(x_min)
        while y_min_up < y_min + 0.5:
            x_up += 1e-5
            y_min_up = func(x_up)
        while y_min_down < y_min + 0.5:
            x_down -= 1e-5
            y_min_down = func(x_down)
        x_p = x_up - x_min
        x_n = x_min - x_down
        return np.around(x_p, decimals=decimals), np.around(x_n,  decimals=decimals)
    else:
        raise ValueError('Unknown method %s' % method)
