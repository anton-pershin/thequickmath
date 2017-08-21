import numpy as np
import matplotlib.pyplot as plt

def get_cpp_res(filename):
    cpp_x_file = open(filename)
    tmp = cpp_x_file.read().split()
    cpp_x = np.zeros((len(tmp),))
    for n in range(len(tmp)):
        cpp_x[n] = np.float64(tmp[n])
    return cpp_x


N = 1000
x = np.zeros((N,), dtype=np.float64)
x[0] = 3./4
r = 3.64
for n in range(1, N):
    x[n] = r*x[n-1] * (1 - x[n-1])

#plt.plot(range(0, N), cpp_x)
cpp_x1 = get_cpp_res('x_1.dat')
cpp_x2 = get_cpp_res('x_2.dat')
cpp_x3 = get_cpp_res('x_3.dat')
plt.plot(range(0, N), np.abs(x - cpp_x1))
#plt.plot(range(0, N), np.abs(cpp_x1 - cpp_x2))
#plt.plot(range(0, N), np.abs(cpp_x2 - cpp_x3))
plt.show()