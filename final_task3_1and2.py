import projB as pb
import numpy as np
import matplotlib.pyplot as plt

# plot normalised histogram of measured D^0 life time data
plt.hist(pb.t_m, bins=100, normed=1, fc=(0, 0, 1, 0.1), label='measured lifetime distribution')

# plot fit function
dt = 0.001
t = np.arange(-2, 9, dt)
sigma = 0.25
tau_l = np.arange(0.3, 0.51, 0.1)
labels = []
for tau_i in tau_l:
    ft = np.empty(t.size)  # create an array holder for fit function
    for i in xrange(t.size):
        ft[i] = pb.f_signal(tau_i, t[i], sigma)  # update using f_signal function
    plt.plot(t, ft)
    labels.append(r'Fit function with $\tau = %s$' % tau_i)
plt.xlim(-1.7, 4)
plt.legend(labels, fontsize=17)
plt.xlabel('t (picoseconds)')
plt.ylabel('counts')
plt.title(r'Histogram of Measured $D^0$ Lifetime')

# verify integral fit function over t is independent of tau and sigma
tau = 0.4
sigma_l = np.arange(0.01, 0.5, 0.05)
integrated_s = []
integrated_tau = []
for sigma_i in sigma_l:
    ft = np.empty(t.size)
    for i in range(t.size):
        ft[i] = pb.f_signal(tau, t[i], sigma_i)
    integrated_s.append(np.sum((ft*dt)))
for tau_i in tau_l:
    ft = np.empty(t.size)
    for i in range(t.size):
        ft[i] = pb.f_signal(tau_i, t[i], sigma)
    integrated_tau.append(np.sum((ft*dt)))
print "integrated fit function over t, but changing sigma value: \n", integrated_s
print "integrated fit function over t, but changing tau value: \n", integrated_tau
print "if each of the above list contains elements that are ~ 1, " \
      "\nthen the integral over t is independent of sigma and tau"

plt.show()
