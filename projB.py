"""Name: projB
Purpose: contains functions necessary for 1D and 2D function minimisation to estimate the life time of meson D^0
Author: Yichen Liu
Created: 23/11/2016

Functions
----------
- read_data: read in .txt file of data in two columns.
-

Notes
-----
This module requires numpy and scipy packages to run, and therefore the functions in this module performs the best when
they are used on numpy.array types of data.
"""
from __future__ import division
import numpy as np
import scipy as sp
from scipy.special import erfc
import numpy.linalg as la


def read_data(filename):
    """Read in .txt file of float type data in two columns. returns each column as a numpy.array.

    Parameters
    ----------
    filename : string
        The name of the file with extension as a string. e.g. 'lifetime.txt'

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

t_m, sigma_m = read_data('lifetime.txt')  # permit direct access data from liftime.txt using projB.t_m, projB.sigma_m


def f_signal(tau, t, sigma):
    return (1/(2*tau)) \
        * sp.exp((sigma**2/(2*tau**2))-(t/tau)) \
        * erfc((1/np.sqrt(2))*((sigma/tau) - (t/sigma)))


def NLL(tau, time_list=t_m, sigma_list=sigma_m):
    return np.sum(-np.log(f_signal(tau, time_list, sigma_list)))


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


def fmbackground(t, sigma):
    return sp.exp(-0.5*(t**2/sigma**2))/(sigma*np.sqrt(2*np.pi))


def ft(tau, signal_fraction, t, sigma):
    return signal_fraction * f_signal(tau, t, sigma) + (1 - signal_fraction) * fmbackground(t, sigma)


def curvature(function_array):
    f1 = np.gradient(function_array)
    f2 = np.gradient(f1)
    return np.abs(f2/((1+f1**2)**1.5))


def NLL2D(tau, signal_fraction, t=t_m, sigma=sigma_m):
    if signal_fraction > 1:
        raise ValueError("The signal fraction is only defined between 0 and 1.")
    return np.sum(-np.log(ft(tau, signal_fraction, t, sigma)))


def partial2D(func, x, delta=1e-5):
    """central difference approximation"""
    diff = 2*delta
    return np.array([((func(x[0]+delta, x[1]) - func(x[0]-delta, x[1])) / diff),
                     ((func(x[0], x[1]+delta) - func(x[0], x[1]-delta)) / diff)])


def hessian2D(func, x, delta=1e-4):  # hessian2D based on function calls, 2n+4n^2/2 additional functions calls are needed.
    d2f_dx2 = (-func(x[0] + 2 * delta, x[1]) + 16 * func(x[0] + delta, x[1]) - 30 * func(x[0], x[1])
               + 16 * func(x[0] - delta, x[1]) - func(x[0] - 2 * delta, x[1])) / (12 * delta ** 2)
    d2f_dy2 = (-func(x[0], x[1] + 2 * delta) + 16 * func(x[0], x[1] + delta) - 30 * func(x[0], x[1])
               + 16 * func(x[0], x[1] - delta) - func(x[0], x[1] - 2 * delta)) / (12 * delta ** 2)
    d2f_dxdy = (func(x[0] + delta, x[1] + delta) - func(x[0] + delta, x[1] - delta)
                - func(x[0] - delta, x[1] + delta) + func(x[0] - delta, x[1] - delta)) / (4 * delta ** 2)
    #same as d2f_dydx
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])


def min_gradient_descent(func, x, alpha=1e-6, tol=1e-10, maxiter=1e4):
    niter = 0
    improved = True
    while improved and niter < maxiter:
        niter += 1
        step = alpha*partial2D(func, x)
        x -= step
        print x, step, niter
        if la.norm(step) < tol:
            improved = False
    return x, niter


def min_newton(func, x, tol=1e-10, maxiter=100):
    niter = 0
    improved = True
    while improved and niter < maxiter:
        niter += 1
        H = hessian2D(func, x)
        step = np.dot(-la.inv(H), partial2D(func, x))
        x += step
        if la.norm(step) < tol:
            improved = False
    return x, niter


def min_quasi_newton(func, x, alpha=1e-6, tol=1e-8, maxiter=1e4):
    niter = 0
    improved = True
    G = np.identity(2)
    while improved and niter < maxiter:
        niter += 1
        x_prime = partial2D(func, x)
        step = alpha * np.dot(G, x_prime)
        x -= step
        print x, niter
        if la.norm(step) < tol:
            improved = False
        gamma = partial2D(func, x) - x_prime
        G += np.outer(step, step) / np.dot(gamma, step) \
            - np.dot(np.dot(G, np.outer(step, step)), G) / np.dot(np.dot(gamma, G), gamma)
    return x, niter