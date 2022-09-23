import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import product

def distance(a,b):
    return math.sqrt(sum([(x-y) ** 2 for x,y in zip(a,b)]))

def grid_center(x,y):
    return complex(x / 4 + 1 / 8, y / 4 + 1 / 8)

class Particle:
    def __init__(self, x, y, q):
        self.x = x
        self.y = y
        self.q = q
        self.phi_real = 0
        self.phi_appx = 0

    def __repr__(self):
        return 'Particle(x={}, y={}, q={}, idx={}, real={}, appx={})'.format(self.x, self.y, self.q, self.get_grid_idx(), self.phi_real, self.phi_appx)

    def get_grid_idx(self):
        return (int(self.y * 4), int(self.x * 4))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def get_dict_plain(ps):
    pdict = {}
    for i,p in enumerate(ps):
        grid_idx = p.get_grid_idx()
        if grid_idx in pdict:
            pdict[grid_idx].append((i,1.0))
        else:
            pdict[grid_idx] = [(i,1.0)]
    return pdict

P = 2
ps = [Particle(0.76,0.249,10), Particle(0.1,0.65,-10)]
phi_real = []
phi_appx = []
r = range(20)
for _ in r:
    for p_t in ps:
        s = 0
        z = complex(p_t.x, p_t.y)
        for p_s in ps:
            zi = complex(p_s.x, p_s.y)
            if p_t != p_s:
                s += -p_s.q * cmath.log(z - zi)
        p_t.phi_real = s.real

    pdict = get_dict_plain(ps)

    M = np.zeros([P+1,4,4], dtype=np.complex)
    Q = np.zeros([4,4])
    L = np.zeros([P+1,4,4], dtype=np.complex)

    # P2M
    for gidx_x, gidx_y in product(range(4), range(4)):
        gidx = (gidx_y, gidx_x)
        if gidx in pdict:
            for pidx,w in pdict[gidx]:
                p = ps[pidx]
                Q[gidx] += p.q
                for k in range(1,1+P+1):
                    zi = complex(p.x, p.y)
                    zd = grid_center(gidx_x, gidx_y)
                    M[(k-1, gidx_y, gidx_x)] += w * -p.q * ((zi - zd) ** k) / k

    # M2L
    for tidx_x, tidx_y in product(range(4), range(4)):
        tidx = (tidx_x, tidx_y)
        wpp = grid_center(tidx_x, tidx_y)
        for sidx_x, sidx_y in product(range(4), range(4)):
            sidx = (sidx_x, sidx_y)
            zpp = grid_center(sidx_x, sidx_y)
            if abs(tidx_y - sidx_y) > 1 or abs(tidx_x - sidx_x) > 1:
                c0 = sum(M[(k-1, sidx_y, sidx_x)] / ((zpp - wpp) ** k) * (-1) ** k for k in range(1, 1+P+1))
                L[0, tidx_y, tidx_x] += Q[(sidx_y, sidx_x)] * cmath.log(grid_center(tidx_x, tidx_y) - grid_center(sidx_x, sidx_y)) + c0
                for t in range(1,1+P):
                    L[t, tidx_y, tidx_x] += -Q[(sidx_y, sidx_x)] / (t * ((zpp - wpp) ** t)) + 1 / ((zpp - wpp) ** t) * sum(M[(l-1, sidx_y, sidx_x)] / ((zpp - wpp) ** l) * comb(t + l - 1, l - 1) * (-1) ** l for l in range(1, P+1))

    # L2P
    for widx,v in pdict.items():
        wpp = grid_center(widx[1], widx[0])
        for zidx,_ in v:
            z = complex(ps[zidx].x, ps[zidx].y)
            ps[zidx].phi_appx = -sum(((z - wpp) ** t) * L[(t, widx[0], widx[1])] for t in range(0, 1+P)).real

    print(ps[0])
    phi_real.append(ps[0].phi_real)
    phi_appx.append(ps[0].phi_appx)
    ps[0].y += 0.0001

figure, axes = plt.subplots()
axes.plot(r, phi_real, label='real')  
axes.plot(r, phi_appx, label='appx')
plt.ylabel('potential')
axes.legend()
#axes.set_ylim([-2.65, -2.55])

plt.show()
