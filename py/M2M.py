import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def random_complex():
    l = random.uniform(0,1)
    theta = random.uniform(-math.pi,math.pi)
    return complex(l * math.cos(theta), l * math.sin(theta))

random.seed(314)

N = 1000
zd = -2 + 3j
zdd = -3 + 1j
z = 6 + 3j
zis = [zd + random_complex() for _ in range(N)]
qis = [random.uniform(0,1) for _ in range(N)]

phi_real = sum([qi * cmath.log(z - zi) for qi,zi in zip(qis,zis)])
print(f'phi_real = {phi_real}')

errs = []
Ps = range(0,11)
for P in Ps:
    Qs = sum(qis)
    aks = [sum([-qi * ((zi - zd) ** k) / k for qi,zi in zip(qis,zis)]) for k in range(1,P+1)]
    bls = [-Qs * ((zd - zdd) ** l) / l + sum([comb(l - 1, k - 1) * aks[k - 1] * (zd - zdd) ** (l - k) for k in range(1,l+1)]) for l in range(1,P+1)]
    phi_appx = Qs * cmath.log(z - zdd) + sum(bls[l - 1] / (z - zdd) ** l for l in range(1,P+1))  
    err = abs((phi_appx - phi_real).real) / abs(phi_real.real)
    errs.append(err)
    print(f'phi_appx = {phi_appx} err = {err}')

plt.plot(Ps, errs, '-o')
plt.xlabel('P')
plt.ylabel('realtive_err')
#plt.show()


figure, axes = plt.subplots()
z_circle = plt.Circle((z.real, z.imag), 0.1, fill=True)
zd_circle = plt.Circle((zd.real, zd.imag), 1, fill=False)
zdd_circle = plt.Circle((zdd.real, zdd.imag), cmath.polar(zd - zdd)[0] + 1, fill=False)

axes.set_aspect(1)
axes.add_patch(z_circle)
axes.add_patch(zd_circle)
axes.add_patch(zdd_circle)
axes.set_xlim([-5, 3])
axes.set_ylim([-2, 6])

plt.show()
