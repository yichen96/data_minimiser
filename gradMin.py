import projB as pb
import numpy.linalg as la
import numpy as np
#from matplotlib import pyplot as plt
tau_l = np.arange(0.1,0.8,0.001)
signalfrac_l = np.arange(0.8,1.01,0.1)
X, Y = np.meshgrid(tau_l, signalfrac_l)
Z = np.ndarray(shape=(signalfrac_l.size,tau_l.size))
P = np.ndarray(shape=(signalfrac_l.size,tau_l.size))
for i in range(signalfrac_l.size):
    for j in range(tau_l.size):
        Z[i][j] = pb.NLL2D(tau_l[j], signalfrac_l[i])
# print signalfrac_l
# print len(Z)
# print len(Z[0])
# print min(Z[-1])

def partial2D(func, x, delta=1e-2):
    """central difference approximation"""
    diff = 2*delta
    return np.array([((func(x[0]+delta, x[1]) - func(x[0]-delta, x[1])) / diff),
                     ((func(x[0], x[1]+delta) - func(x[0], x[1]-delta)) / diff)])

def parabol(x,y):
    return x**2 + y**2

print partial2D(parabol, np.array([3.,5.]))


def gradientMin(func, x, alpha=1e-6, tol=1e-16, maxiter=1e5):
    niter = 0
    improved = True
    while improved and niter < maxiter:
        niter += 1
        step = alpha*partial2D(func, x)
        x -= step
        if la.norm(step) < tol:
            improved = False
    return x, niter

start = np.array([0.4,0.4])
print gradientMin(pb.NLL2D, start)
