import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt

def random_complex():
    l = random.uniform(0,1)
    theta = random.uniform(-math.pi,math.pi)
    return complex(l * math.cos(theta), l * math.sin(theta))

random.seed(314)

N = 100
zd = -2 + 3j
z = 1 + 2j
zis = [zd + random_complex() for _ in range(N)]
qis = [random.uniform(0,1) for _ in range(N)]

phi_real = sum([-qi * cmath.log(z - zi) for qi,zi in zip(qis,zis)])
print(f'phi_real = {phi_real}')
f_real = sum([qi * 1 / (z - zi) for qi,zi in zip(qis, zis)])
f_real_xy = (f_real.real, -f_real.imag)
print(f'f_real = {f_real_xy}')

err_phis = []
err_fs = []
Ps = range(0,11)
for P in Ps:
    Qs = sum(qis)
    aks = [sum([-qi * ((zi - zd) ** k) / k for qi,zi in zip(qis,zis)]) for k in range(1,P+1)]  
    phi_appx = Qs * (-cmath.log(z - zd)) - sum(aks[k - 1] * 1 / (z - zd) ** k for k in range(1,P+1))  
    err_phi = abs((phi_appx - phi_real).real) / abs(phi_real.real)
    f_appx = Qs * (1 / (z - zd)) - sum(aks[k - 1] * k * (z - zd) ** (-k - 1) for k in range(1, P+1))
    err_f = abs((f_appx - f_real).real) / abs(f_real.real)
    f_appx_xy = (f_appx.real, -f_appx.imag)
    err_phis.append(err_phi)
    err_fs.append(err_f)
    print(f'phi_appx = {phi_appx} err = {err_phi},  f_appx = {f_appx_xy} err = {err_f}')

plt.plot(Ps, err_phis, '-o', label = 'potential')
plt.plot(Ps, err_fs, '-o', label = 'force')
plt.xlabel('P')
plt.ylabel('realtive_err')
plt.legend()
plt.show()
