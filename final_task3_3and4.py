import projB as pb
import numpy as np
import matplotlib.pyplot as plt

# plot the NLL of f_signal fit function
tau_l = np.arange(0.2, 0.7, 0.001)
NLL = np.zeros(tau_l.size)  # create an array holder of NLL

for i in xrange(len(tau_l)):
    NLL[i] = pb.nll1d(tau_l[i])  # update using nll1d function

plt.plot(tau_l, NLL, label='Negative log likelihood')
plt.xlabel(r'$\tau$ (picosecond)', fontsize=16)
plt.ylabel('NLL', fontsize=16)
plt.title('Negative Log Likelihood of the Fit Function')

# minimise NLL
min_tau,  min_NLL, last_xlist, last_ylist = pb.min_parabolic(pb.nll1d, [0.3, 0.4, 0.5], tol=1e-5)
print "using parabolic minimiser, the best fitted tau is %.4f picoseconds" % min_tau

plt.plot(min_tau, min_NLL, 'ro', label='NLL minimum', markersize=11)
plt.annotate('minimum at\n(%.4f, %.0f)' % (min_tau, min_NLL), xy=(0.36, 6350), fontsize=16)
plt.legend(numpoints=1, fontsize=20)

plt.show()
