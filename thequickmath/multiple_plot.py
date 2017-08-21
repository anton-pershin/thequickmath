from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import thequickmath as qm

x = np.linspace(-10, 10., num=1000, endpoint=True)
func_1 = lambda h, x: h/x
func_2 = lambda r, x: r/np.sqrt(1 + x**2)

#for i in np.arange(-1, 1, 0.1):
#    plt.plot(x, func_1(i, x))

#for i in np.arange(0, 2, 0.1):
#    plt.plot(x, func_2(i, x))

#plt.plot(x, func_1(i, x))
plt.plot(x, func_2(1, x))

#plt.axis([0, 1, -0.2, 0.001])
#plt.xlabel('$s$')
#plt.ylabel('$f_k(1 - s Ma^{-\gamma})$')
plt.show()