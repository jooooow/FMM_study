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

N = 1000
zd = -2 + 3j
z = 1 + 2j
zis = [zd + random_complex() for _ in range(N)]
qis = [random.uniform(0,1) for _ in range(N)]

phi_real = sum([qi * cmath.log(z - zi) for qi,zi in zip(qis,zis)])
print(f'phi_real = {phi_real}')

errs = []
Ps = range(0,11)
for P in Ps:
    Qs = sum(qis)
    aks = [sum([-qi * ((zi - zd) ** k) / k for qi,zi in zip(qis,zis)]) for k in range(1,P+1)]  
    phi_appx = Qs * cmath.log(z - zd) + sum(aks[k - 1] / (z - zd) ** k for k in range(1,P+1))  
    err = abs((phi_appx - phi_real).real) / abs(phi_real.real)
    errs.append(err)
    print(f'phi_appx = {phi_appx} err = {err}')

plt.plot(Ps, errs, '-o')
plt.xlabel('P')
plt.ylabel('realtive_err')
plt.show()
