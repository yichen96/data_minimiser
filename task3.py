from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
import projB as pb

t_test = np.arange(-2, 9, 0.001)
sigma_test = 0.2
tau_l = np.arange(0.1, 1, 0.0001)
NLL = np.zeros(tau_l.size)

for i in xrange(len(tau_l)):
    NLL[i] = pb.NLL(tau_l[i])

# plt.plot(tau_l, NLL)
# plt.xlabel('tau')
# plt.ylabel('NLL(tau)')
# plt.show()

# min_tau,  min_NLL, final_xlist, final_ylist = pb.min_parabolic(pb.NLL, [0.2, 0.4, 0.5])
# print 'mintau',float('%.4g'%min_tau)
# tau_l = np.around(tau_l, decimals=4)
# NLL = np.around(NLL, decimals=4)
# i = np.where(tau_l == float('%.4g'%min_tau))
# i = i[0][0]
# print i
# curvature = pb.curvature(NLL)
# derivative = pb.derivative2(NLL)
# print 'curvature is', curvature[i]
# print '2ndderivative', derivative[i]
# changed = np.around(min_NLL, decimals=4) + 0.5
# print changed


def findupper(array, starti, value):
    n = array[starti]
    while n < value:
        starti += 1
        n = array[starti]
    return starti, n

def findlower(array, starti, value):
    n = array[starti]
    while n < value:
        starti -= 1
        n = array[starti]
    return starti, n

j, NLLp = findupper(NLL, i, changed)
print j, tau_l[j]
k, NLLm = findlower(NLL, i, changed)
print k, tau_l[k]

'''
min tau = 0.4045
curvature = 0.0005

##currently all in picosecond
NLL+0.5 --> taup = 0.4093 = tau_min + 0.0048
--> taum = 0.3398 = tau_min - 0.0047'''

# def Parabole(x, xlist, ylist):
#     return ((x-xlist[1])*(x-xlist[2])/(xlist[0]-xlist[1])*(xlist[0]-xlist[2]))*ylist[0] \
#     + ((x-xlist[0])*(x-xlist[2])/(xlist[1]-xlist[0])*(xlist[1]-xlist[2]))*ylist[1] \
#     + ((x - xlist[0]) * (x - xlist[1]) / (xlist[2] - xlist[0]) * (xlist[2] - xlist[1])) * ylist[2]
#
# x = np.arange(-3,3,0.0001)
# parabole = Parabole(x, final_xlist, final_ylist)
# min_para = min(parabole)
# derivative_2 = pb.derivative2(parabole)
# min_deri = derivative_2[np.where(parabole == min(parabole))]
# print min_deri
# print np.sqrt(1/(4*min_deri))
# print final_xlist

def curv(x,y):
    d = (x[1]-x[0])*(x[2]-x[0])*(x[2]-x[1])
    return (x[2]-x[1])*y[0]/d + (x[0]-x[2])*y[1]/d + (x[1]-x[0])*y[2]/d

a = curv(final_xlist, final_ylist)
print a
print np.sqrt(1/(2*a))
