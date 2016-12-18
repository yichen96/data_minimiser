import projB as pb
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats

# find accuracy of fit result (from using parabolic method minimiser)
start_search = [0.3, 0.4, 0.5]
error_curvature = pb.std_error(pb.nll1d, start_search, method='curvature')
error_p, error_n = pb.std_error(pb.nll1d, start_search, method='scan', decimals=5)
print 'standard deviation found using curvature of the estimated parabola is %.4f picoseconds' % error_curvature
print 'standard deviation found scanning the NLL function are:' \
      '\n%.5f picoseconds on the left hand side of the minimum,' \
      '\nand %.5f picoseconds on the right hand side of the minimum;' \
      '\napproximated to 2 significant number,' \
      '\nrespectively are %.4f and %.4f picoseconds.' % (error_n, error_p, error_n, error_p)
print 'std deviation found using both method agree with each other.'

# how many measurement would you need to get accuracy of 10^-15 seconds?
time_list = copy.copy(pb.t_m)
sigma_list = copy.copy(pb.sigma_m)
truncation = 100
n = []
error = []

for i in xrange(1000, time_list.size + 1, truncation):
    t = copy.copy(time_list[:i])
    s = copy.copy(sigma_list[:i])


    def nll(tau, time=t, sigma=s):  # define a nll function each time taking the truncated t and s arrays
        L = np.sum(-np.log(pb.f_signal(tau, time, sigma)))
        if np.isnan(L):
            raise ValueError('Result is not a number')
        return L

    n.append(len(t))
    e = pb.std_error(nll, [0.38, 0.4, 0.42])
    error.append(e)

# take the natural logarithm of data to get a straight line
ln_n = np.log(n)
ln_e = np.log(error)

# create the fitting line
slope, intercept, r_value, p_value, std_err = stats.linregress(ln_n, ln_e)
x_i = np.arange(4.5, 14, 0.1)
line = slope*x_i+intercept

# calculate the number of measurements needed from the fitting line
ln_want = np.log(1e-3)  # the standard deviation that we want to achieve
ln_result = (ln_want - intercept) / slope
result = np.exp(ln_result)
print 'to get accuracy of 10^-15 seconds, %.1e measurements would be needed' \
      ' (approximated to 2 significant number).' % result


# plot the graph of data and fitting
plt.plot(ln_n, ln_e, 'b.', label=r'$\ln{\sigma}$')
plt.plot(x_i, line, 'r-', label=r'Linear Fit of $\ln{\sigma}$')
plt.xlabel(r'$\ln{n}$', fontsize=16)
plt.ylabel(r'$\ln{\sigma}$', fontsize=16)
plt.text(7.6, -4.5, r'$\ln{\sigma} = %.2f \ln{n} %.2f $' % (slope, intercept), fontsize=19)
plt.xlim(6.8, 9.5)
plt.ylim(-5.5, -4)
plt.legend(numpoints=1, fontsize=20)
plt.title(r'Standard Deviation of the Average Lifetime Reduced With More Measurements')
plt.show()
