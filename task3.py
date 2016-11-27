from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
import projB as pb

t_test = np.arange(-2, 9, 0.001)
sigma_test = 0.2
tau_l = np.arange(0.1, 2, 0.0001)
NLL = np.zeros(tau_l.size)

for i in xrange(len(tau_l)):
    NLL[i] = pb.NLL(tau_l[i])

# plt.plot(tau_l, NLL)
# plt.xlabel('tau')
# plt.ylabel('NLL(tau)')
# plt.show()

min_tau,  min_NLL, final_list = pb.parabolicMin(pb.NLL, [0.2, 0.4, 0.5])

## to find curvature at minimum
i = np.where(NLL==min(NLL))
print tau_l[i]
curvature = pb.curvature(NLL)
print curvature[i]

