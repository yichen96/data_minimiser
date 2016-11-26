from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import projB as pb
import scipy.integrate as si

plt.hist(pb.t_m, bins=100, normed=1, fc=(0, 0, 1, 0.1))

t_test = np.arange(-2, 9, 0.001)
sigma_test = 0.15
#tau_l = np.arange(0.48,0.68,0.02)
tau_l = np.arange(0.4, 0.6, 0.02)
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(tau_l))])
labels = []
for tau_i in tau_l:
    fmt = np.empty(t_test.size)
    for i in xrange(t_test.size):
        fmt[i] = pb.fm2(tau_i, t_test[i], sigma_test)
    plt.plot(t_test, fmt)
    labels.append(r'$\tau = %s$' % tau_i)
plt.xlim(-2, 6)
plt.legend(labels)
plt.show()
#0.52 is the best??

tau = 0.52
# check if integral independent of sigma
sigma_l = np.arange(0.01, 0.5, 0.05)
for sigma in sigma_l:
    fmt = np.empty(t_test.size)
    for i in xrange(t_test.size):
        fmt[i] = pb.fm2(tau, t_test[i], sigma)
    print np.sum((fmt*0.001))
