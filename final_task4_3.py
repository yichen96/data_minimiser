import projB as pb
import numpy as np
import matplotlib.pyplot as plt

tau_l = np.arange(0.4, 0.53, 0.01)
signalfrac_l = np.arange(0.38, 1.01, 0.01)  # signal fraction is only defined between 0 and 1

Y, X = np.meshgrid(signalfrac_l, tau_l)
Z = np.ndarray(shape=(tau_l.size, signalfrac_l.size))
for i in range(tau_l.size):
    for j in range(signalfrac_l.size):
        Z[i][j] = pb.nll2d(tau_l[i], signalfrac_l[j])

CS = plt.contour(X, Y, Z, 8)
plt.clabel(CS, inline=1)

startpoint = np.array([0.43, 0.4])
plt.plot(startpoint[0], startpoint[1], 'ko')
minimum_grad, iter_grad, path_grad = pb.min_gradient_descent(pb.nll2d, startpoint, alpha=2e-5)
print 'Using gradient method, in %g iterations, ' \
      'minimum is found when tau = %.4f and signal_fraction = %.4f' \
      % (iter_grad, minimum_grad[0], minimum_grad[1])

minimum_newton, iter_newton, path_newton = pb.min_newton(pb.nll2d, startpoint, bound_mode=False)
print 'Using newton method, in %g iterations, ' \
      'minimum is found when tau = %.4f and signal_fraction = %.4f' \
      % (iter_newton, minimum_newton[0], minimum_newton[1])

minimum_quasi, iter_quasi, path_quasi = pb.min_gradient_descent(pb.nll2d, startpoint, alpha=2e-5)
print 'Using quasi-newton method, in %g iterations, ' \
      'minimum is found when tau = %.4f and signal_fraction = %.4f' \
      % (iter_quasi, minimum_quasi[0], minimum_quasi[1])
plt.plot(path_grad[:, 0], path_grad[:, 1], 'ro-', label='gradient method')
plt.plot(path_newton[:, 0], path_newton[:, 1], 'go-', label='newton method')
plt.plot(startpoint[0], startpoint[1], 'k*', markersize=18, label='start point')
plt.ylim(0.38, 1.064)
plt.ylabel('signal fraction', fontsize=16)
plt.xlabel(r'$\tau$ (picosecond)', fontsize=16)
plt.legend(numpoints=1, loc='upper center', ncol=3, mode="expand", fontsize=14)
plt.title('Different Minimiser Converge to the Same Point')
plt.show()
