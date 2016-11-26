from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import projB as pb

t_test = np.arange(-2, 9, 0.001)
sigma_test = 0.2
tau_l = np.arange(0.1, 3, 0.001)
NLL = []

for tau in tau_l:
    NLL.append(pb.NLL(tau, pb.t_m, pb.sigma_m))

plt.plot(tau_l, NLL)
plt.xlabel('tau')
plt.ylabel('NLL(tau)')
plt.show()
