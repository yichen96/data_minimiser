'''demonstrating contour
and what X, Y, shape is for Z'''
import projB as pb
import numpy as np
from matplotlib import pyplot as plt
tau_l = np.arange(0.1,0.8,0.01)
signalfrac_l = np.arange(0.1,1,0.01)
X, Y = np.meshgrid(tau_l, signalfrac_l)
Z = np.ndarray(shape=(signalfrac_l.size,tau_l.size))
for i in range(signalfrac_l.size):
    for j in range(tau_l.size):
        Z[i][j] = pb.NLL2D(tau_l[j], signalfrac_l[i])

plt.contour(X, Y, Z,10)
plt.show()