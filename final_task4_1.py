import projB as pb
import numpy as np
import matplotlib.pyplot as plt

tau_l = np.arange(0.2, 0.7001, 0.01)
signalfrac_l = np.arange(0.1, 1.0001, 0.01)  # signal fraction is only defined between 0 and 1
Y, X = np.meshgrid(signalfrac_l, tau_l)
Z = np.ndarray(shape=(tau_l.size, signalfrac_l.size))
for i in range(tau_l.size):
    for j in range(signalfrac_l.size):
        Z[i][j] = pb.NLL2D(tau_l[i], signalfrac_l[j])

CS = plt.contourf(X, Y, Z, 20)
plt.xlim(0.2, 0.7)
plt.ylim(0.1, 1)
plt.xlabel(r'$\tau$ (picosecond)', fontsize=16)
plt.ylabel('signal fraction', fontsize=16)
plt.title('Contour of Negative Log Likelihood Comprising Background Noise')
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('Negative Log Likelihood')
plt.show()
