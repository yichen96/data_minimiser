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

min_tau,  min_NLL, final_list = pb.parabolicMin(pb.NLL, [0.2, 0.4, 0.5])
print 'mintau',float('%.4g'%min_tau)
tau_l = np.around(tau_l, decimals=4)
NLL = np.around(NLL, decimals=4)
i = np.where(tau_l == float('%.4g'%min_tau))
i = i[0][0]
print i
curvature = pb.curvature(NLL)
print 'curvature is', curvature[i]
changed = np.around(min_NLL, decimals=4) + 0.5
print changed


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

NLL+0.5 --> taup = 0.4093 = tau_min + 0.0048
--> taum = 0.3398 = tau_min - 0.0047'''


