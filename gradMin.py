import projB as pb
import numpy.linalg as la
import numpy as np
#from matplotlib import pyplot as plt
# tau_l = np.arange(0.1,0.8,0.001)
# signalfrac_l = np.arange(0.1,1.001,0.01)
# X, Y = np.meshgrid(tau_l, signalfrac_l)
# Z = np.ndarray(shape=(signalfrac_l.size,tau_l.size))
# P = np.ndarray(shape=(signalfrac_l.size,tau_l.size))
# for i in range(signalfrac_l.size):
#     for j in range(tau_l.size):
#         Z[i][j] = pb.NLL2D(tau_l[j], signalfrac_l[i])
# print signalfrac_l
# print len(Z)
# print len(Z[0])
# print min(Z[-1])

# def parabol(x,y):
#     return x**2 + y**2
#
# print _partial2D(parabol, np.array([3.,5.]))

# # print min_gradient_descent(pb.NLL2D, start)

# print newtonMin(pb.NLL2D, start)
#
# NLL2d = np.empty(signalfrac_l.size)
# for i in xrange(len(signalfrac_l)):
#     NLL2d[i] = pb.NLL2D(0.4, signalfrac_l[i])
# print grad2(NLL2d)
# x_prime = pb._partial2D(pb.NLL2D, np.array([0.4, 0.8]))
# print x_prime
# x_prime_p = pb._partial2D(pb.NLL2D, x_prime)
# print x_prime_p


# def min_quasi_newton(func, x, alpha=1e-6, tol=1e-10, maxiter=1e4):
#     niter = 0
#     improved = True
#     G = np.identity(2)
#     print x, niter
#     while improved and niter < maxiter:
#         niter += 1
#         x_prime = pb._partial2D(func, x)
#         step = alpha * np.dot(G, x_prime)
#         x -= step
#         if la.norm(step) < tol:
#             improved = False
#         gamma = pb._partial2D(func, x) - x_prime
#         G += np.outer(step, step) / np.dot(gamma, step) \
#             - np.dot(np.dot(G, np.outer(step, step)), G) / np.dot(np.dot(gamma, G), gamma)
#     return x, niter


# start1 = np.array([0.55, 0.7])
# start2 = np.array([0.3, 0.9])
# x2, n2 = pb.min_quasi_newton(pb.NLL2D, start2, alpha=1e-5, tol=1e-8, maxiter=1e4)
# x, n = pb.min_gradient_descent(pb.NLL2D, start2, alpha=1e-5, tol=1e-8, maxiter=1e4)

##LOCAL VARIABLE PROBLEM!! why when start are the same the second function inherite all variables from the previous function??

# print start1
# print pb.min_newton(pb.NLL2D, start1)

def f(x, y):
    return (x-2)**2 + (y-2)**2

#print pb.min_gradient_descent(pb.NLL2D, np.array([0.4, 0.8]), alpha=1e-3, tol=1e-8, maxiter=1e4)
print pb.min_newton(pb.NLL2D, np.array([0.8, 0.8]), tol=1e-8, maxiter=1e4)

