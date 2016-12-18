import projB as pb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


min_tau = 0.4097
min_signal_fraction = 0.9837


def nll_constant_s(tau):
    return pb.nll(pb.f_total, tau, min_signal_fraction, pb.t_m, pb.sigma_m)


def nll_constant_t(signal_frac):
    return pb.nll(pb.f_total, min_tau, signal_frac, pb.t_m, pb.sigma_m)

startpoint_tau = [0.4, 0.42, 0.39]
startpoint_sigfrac = [0.98, 0.97, 0.99]
error_tau_p, error_tau_n = pb.std_error(nll_constant_s, startpoint_tau, method='scan')
error_sigfrac_p, error_sigfrac_n = pb.std_error(nll_constant_t, startpoint_sigfrac, method='scan')
print 'the error on tau from scanning left hand side is %.4f,' \
      '\nfrom scanning right hand side is %.4f;' \
      '\nthe error on signal_fraction from scanning left hand side is %.4f,' \
      '\nfrom scanning right hand side is %.4f.' \
      % (error_tau_n, error_tau_p, error_sigfrac_n, error_sigfrac_p)

min_NLL = pb.nll2d(min_tau, min_signal_fraction)

tau_l = np.arange(0.4, 0.42, 0.001)
signalfrac_l = np.arange(0.97, 0.993, 0.001)

error_tau_array = np.array([[[min_tau-error_tau_n, min_tau+error_tau_p], [min_signal_fraction, min_signal_fraction]]])
error_sigfrac_array = np.array([[[min_tau, min_tau],
                                 [min_signal_fraction-error_sigfrac_n, min_signal_fraction+error_sigfrac_p]]])

Y, X = np.meshgrid(signalfrac_l, tau_l)
Z = np.ndarray(shape=(tau_l.size, signalfrac_l.size))
for i in range(tau_l.size):
    for j in range(signalfrac_l.size):
        Z[i][j] = pb.nll2d(tau_l[i], signalfrac_l[j])
levels = np.arange(min_NLL, 6218.9, 0.5)
CS = plt.contourf(X, Y, Z, levels=levels, alpha=0.5, colors='g')
plt.clabel(CS, inline=1, fmt='%.1f', fontsize=13)
green_patch = mpatches.Patch(color='green', alpha=0.5, label=r'1 $\sigma$ from the minimum')
black_patch = mpatches.Patch(facecolor='white', edgecolor='black', label=r'1 $\sigma$ delimited by error found from $\tau$ and signal fraction')
minimum, = plt.plot(min_tau, min_signal_fraction, 'ro', label='minimum where NLL = %.1f' % min_NLL)
plt.annotate('minimum at (%.3f, %.3f)' % (min_tau, min_signal_fraction), xy=(0.407, 0.985))

plt.hlines(error_sigfrac_array[0][1][0], error_tau_array[0][0][0], error_tau_array[0][0][1])
plt.hlines(error_sigfrac_array[0][1][1], error_tau_array[0][0][0], error_tau_array[0][0][1])
plt.vlines(error_tau_array[0][0][0], error_sigfrac_array[0][1][0], error_sigfrac_array[0][1][1])
plt.vlines(error_tau_array[0][0][1], error_sigfrac_array[0][1][0], error_sigfrac_array[0][1][1])

plt.xlim(0.4, 0.42)
plt.ylim(0.97, 1)
plt.xlabel(r'$\tau$ (picosecond)', fontsize=16)
plt.ylabel('signal fraction', fontsize=16)
plt.legend(numpoints=1, fontsize=13, handles=[minimum, green_patch, black_patch])
plt.title('Standard Deviation of the Average Lifetime and Signal Fraction')
plt.show()
