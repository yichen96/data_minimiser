import projB as pb
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import stats

time_list = copy.copy(pb.t_m)
sigma_list = copy.copy(pb.sigma_m)
truncation = 100
n = []
error = []

for i in xrange(1000, time_list.size + 1, truncation):
    t = copy.copy(time_list[:i])
    s = copy.copy(sigma_list[:i])


    def nll(tau, time=t, sigma=s):
        L = np.sum(-np.log(pb.f_signal(tau, time, sigma)))
        if np.isnan(L):
            raise ValueError('Result is not a number')
        return L

    n.append(len(t))
    e = pb.std_error(nll, [0.38, 0.4, 0.42])
    error.append(e)

ln_n = np.log(n)
ln_e = np.log(error)
slope, intercept, r_value, p_value, std_err = stats.linregress(ln_n, ln_e)

want = np.log(1e-3)
result = (want - intercept) / slope
print result
x_i = np.arange(4.5, 14, 0.1)
line = slope*x_i+intercept


plt.plot(ln_n, ln_e, 'b.', label=r'$\ln{\sigma}$')
plt.plot(x_i, line, 'r-', label='Linear Fit')
# plt.axhline(y=want)
# plt.plot(result, want, 'ko')
plt.xlabel(r'$\ln{n}$')
plt.ylabel(r'$\ln{\sigma}$')
plt.text(7.6, -4.5, r'$\ln{\sigma} = %.2f \ln{n} %.2f $' % (slope, intercept), fontsize=19)
plt.xlim(6.8, 9.5)
plt.ylim(-5.5, -4)
plt.legend()
plt.show()
