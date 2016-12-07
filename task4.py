'''show how to use the NLL2D and meshgrid'''
import numpy as np
import matplotlib.pyplot as plt
import projB as pb
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
tau_list = np.arange(0.1,1,0.01)
signalfrac_list = np.arange(0.1,1,0.01)
NLL2D = np.ndarray(shape=(tau_list.size,signalfrac_list.size))
X,Y = np.meshgrid(signalfrac_list, tau_list)
print NLL2D
print X
print Y
for i in range(tau_list.size):
    for j in range(signalfrac_list.size):
        NLL2D[i][j] = pb.NLL2D(tau_list[i], signalfrac_list[j])

print'NLL2d', NLL2D
print 'NLL2d0', NLL2D[:][0]
print len(tau_list), len(signalfrac_list), len(NLL2D), len(NLL2D[0])
print len(X), len(Y)
print len(X[0]), len(Y[0])
fig = plt.figure()
ax = fig.gca(projection='3d')


Z = NLL2D
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

plt.show()
