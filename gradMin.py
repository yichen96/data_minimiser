import projB as pb
import numpy.linalg as la
import numpy as np
#from matplotlib import pyplot as plt
# tau_l = np.arange(0.1,0.8,0.001)
signalfrac_l = np.arange(0.1,1.001,0.01)
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
# print partial2D(parabol, np.array([3.,5.]))

# # print min_gradient_descent(pb.NLL2D, start)


def hessian(func, x, delta=1e-4):  # hessian2D based on function calls, 2n+4n^2/2 additional functions calls are needed.
    d2f_dx2 = (-func(x[0] + 2 * delta, x[1]) + 16 * func(x[0] + delta, x[1]) - 30 * func(x[0], x[1])
               + 16 * func(x[0] - delta, x[1]) - func(x[0] - 2 * delta, x[1])) / (12 * delta ** 2)
    d2f_dy2 = (-func(x[0], x[1] + 2 * delta) + 16 * func(x[0], x[1] + delta) - 30 * func(x[0], x[1])
               + 16 * func(x[0], x[1] - delta) - func(x[0], x[1] - 2 * delta)) / (12 * delta ** 2)
    d2f_dxdy = (func(x[0] + delta, x[1] + delta) - func(x[0] + delta, x[1] - delta)
                - func(x[0] - delta, x[1] + delta) + func(x[0] - delta, x[1] - delta)) / (4 * delta ** 2)
    #same as d2f_dydx
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])



start = np.array([0.5,0.8])
def newtonMin(func, x, tol=1e-10, maxiter=100):
    niter = 0
    improved = True
    while improved and niter < maxiter:
        niter += 1
        H = hessian(func, x)
        d = np.dot(-la.inv(H), pb.partial2D(func, x))
        x += d
        if la.norm(d) < tol:
            improved = False
    return x, niter
print newtonMin(pb.NLL2D, start)
#
# def grad2(array):
#     return np.gradient(np.gradient(array))
#
# NLL2d = np.empty(signalfrac_l.size)
# for i in xrange(len(signalfrac_l)):
#     NLL2d[i] = pb.NLL2D(0.4, signalfrac_l[i])
# print grad2(NLL2d)
# x_prime = pb.partial2D(pb.NLL2D, np.array([0.4, 0.8]))
# print x_prime
# x_prime_p = pb.partial2D(pb.NLL2D, x_prime)
# print x_prime_p