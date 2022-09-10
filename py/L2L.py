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
wdd = 3 + 5.5j
wd = 2.5 + 3.2j
z = 2 + 3j
ddr = 3.5
zis = [zd + random_complex() for _ in range(N)]
qis = [random.uniform(0,1) for _ in range(N)]

phi_real = sum([qi * cmath.log(z - zi) for qi,zi in zip(qis,zis)])
print(f'phi_real = {phi_real}')

errs = []
Ps = range(0,31)
for P in Ps:
    Qs = sum(qis)
    aks = [sum([-qi * ((zi - zd) ** k) / k for qi,zi in zip(qis,zis)]) for k in range(1,P+1)]
    bls = [-Qs * ((zd - zdd) ** l) / l + sum([comb(l - 1, k - 1) * aks[k - 1] * (zd - zdd) ** (l - k) for k in range(1,l+1)]) for l in range(1,P+1)]
    hts = [-Qs / (t * (zdd - wdd) ** t) + (1 / (zdd - wdd) ** t) * sum(bls[l - 1] / ((zdd - wdd) ** l) * comb(t+l-1, l-1) * (-1) ** l for l in range(1, P + 1)) for t in range(1, P+1)]
    c0 = sum(bls[l-1] / ((zdd - wdd) ** l) * (-1) ** l for l in range(1, P+1))
    hts = [c0 + Qs * cmath.log(wdd - zdd)] + hts
    uks = [sum(comb(t, k) * ((wd - wdd) ** (t-k)) * hts[t] for t in range(k,P+1)) for k in range(0,P+1)]
    phi_appx = sum(((z - wd) ** k) * uks[k] for k in range(0, P+1))
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
zdd_circle = plt.Circle((zdd.real, zdd.imag), ddr, fill=False)
wdd_circle = plt.Circle((wdd.real, wdd.imag), ddr, fill=False)
wd_circle = plt.Circle((wd.real, wd.imag), 1, fill=False)

axes.set_aspect(1)
axes.add_patch(z_circle)
axes.add_patch(zd_circle)
axes.add_patch(zdd_circle)
axes.add_patch(wdd_circle)
axes.add_patch(wd_circle)
axes.set_xlim([-8, 7])
axes.set_ylim([-5, 10])

plt.show()
