# simulation of molecular dynamics
# harmonic potential for chemical bond
# LJ potential and Coulomb potential for intermolecular interaction

import io
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

random.seed(110)
atom_size = 0.05
region_size = 3
k = 10000
r0 = 0.1
epsilon = 1
sigma = 0.2
rc = 3 * sigma
ke = 8.988e+0
epsilon0 = 8.854e-3
m = 1

class Atom:
    def __init__(self, q, pos, v):
        self.q = q
        self.pos = pos
        self.force = complex(0)
        self.acc = complex(0)
        self.v = v

class Dipole:
    def __init__(self, pos, l, theta):
        self.pos = pos
        v = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        #v = complex(0)
        self.ap = Atom(1, pos + complex(l / 2 * math.cos(theta), l / 2 * math.sin(theta)), v)
        self.an = Atom(-1, pos - complex(l / 2 * math.cos(theta), l / 2 * math.sin(theta)), v)
    def get_bond_force(self):
        r = self.ap.pos - self.an.pos
        r = warp(r)
        delta = r / abs(r)
        fp = -k * (abs(r) - r0) * delta
        fn = k * (abs(r) - r0) * delta
        return [fp, fn]

def GetAtomCircle(atom):
    if atom.q > 0:
        return plt.Circle((atom.pos.real, atom.pos.imag), atom_size, color = 'r')
    elif atom.q < 0:
        return plt.Circle((atom.pos.real, atom.pos.imag), atom_size, color = 'b')
    return plt.Circle((atom.pos.real, atom.pos.imag), atom_size, fill = False)

def GetFrame():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    dst = cv2.imdecode(enc, 1)
    dst = dst[:,:,::-1]
    return dst

def LJ_force(r, epsilon, sigma):
    return 24 * epsilon * (2 * (sigma ** 12) / (r ** 13) - (sigma ** 6) / (r ** 7))

def truncated_LT_force(r, rc, epsilon, sigma):
    if r <= rc:
        return LJ_force(r, epsilon, sigma) - LJ_force(rc, epsilon, sigma)
    return 0

def coulomb_force(a1, a2):
    delta = a1.pos - a2.pos
    delta = warp(delta)
    #f = ke * a1.q * a2.q / (abs(delta) ** 2)
    f = a1.q * a2.q / abs(delta) / (2 * math.pi * epsilon0)
    #f = a1.q * a2.q / abs(delta) / 2
    delta /= abs(delta)
    return f * delta

def warp(x):
    r = x
    if r.real > 0.5 * region_size:
        r -= complex(region_size,0)
    elif r.real < -0.5 * region_size:
        r += complex(region_size,0)
    if r.imag > 0.5 * region_size:
        r -= complex(0,region_size)
    elif r.imag < -0.5 * region_size:
        r += complex(0,region_size)
    return r

fig, ax = plt.subplots()
anime = []
#ds = [Dipole(complex(random.uniform(-0.8,0.8),random.uniform(-0.8,0.8)), 0.1, random.uniform(0, 2 * math.pi))
#      for x in range(10)]
ds = []
N = 5
M = 1.2
for x in np.linspace(-M,M,N):
    for y in np.linspace(-M,M,N):
        #ds.append(Dipole(complex(x,y), 0.1, random.uniform(0, 2 * math.pi)))
        ds.append(Dipole(complex(x,y), 0.1, 0))

dt = 0.005
for i in range(500):
    t = dt * i
    print(f't = {t}')
    plt.cla()

    # force
    for d_tar in ds:
        d_tar.ap.force = 0
        d_tar.an.force = 0
        for d_src in ds:
            if d_tar == d_src:
                continue
            r = d_tar.pos - d_src.pos
            r = warp(r)
            direction = r / abs(r)
            f_LJ = truncated_LT_force(abs(r), rc, epsilon, sigma) * direction
            d_tar.ap.force += f_LJ
            d_tar.an.force += f_LJ

            f_c_pp = coulomb_force(d_tar.ap, d_src.ap)
            f_c_pn = coulomb_force(d_tar.ap, d_src.an)
            f_c_np = coulomb_force(d_tar.an, d_src.ap)
            f_c_nn = coulomb_force(d_tar.an, d_src.an)
            d_tar.ap.force += f_c_pp + f_c_pn
            d_tar.an.force += f_c_np + f_c_nn
            
        f_bond_p, f_bond_n = d_tar.get_bond_force()
        d_tar.ap.force += f_bond_p
        d_tar.an.force += f_bond_n

    # acc, v, pos
    for dipole in ds:
        dipole.ap.acc = dipole.ap.force / m
        dipole.an.acc = dipole.an.force / m
        dipole.ap.v += dipole.ap.acc * dt
        dipole.an.v += dipole.an.acc * dt
        dipole.ap.pos += dipole.ap.v * dt
        dipole.an.pos += dipole.an.v * dt
        dipole.ap.pos = warp(dipole.ap.pos)
        dipole.an.pos = warp(dipole.an.pos)
        #dipole.pos = 0.5 * (dipole.ap.pos + dipole.an.pos)
        dipole.pos = dipole.ap.pos + 0.5 * warp(dipole.an.pos - dipole.ap.pos)

    # visualization
    for dipole in ds:
        ax.add_artist(GetAtomCircle(dipole.ap))
        ax.add_artist(GetAtomCircle(dipole.an))
        p1 = dipole.ap.pos + 0.5 * warp(dipole.an.pos - dipole.ap.pos)
        p2 = dipole.an.pos + 0.5 * warp(dipole.ap.pos - dipole.an.pos)
        ax.plot([dipole.ap.pos.real, p1.real],[dipole.ap.pos.imag, p1.imag], zorder=0, color='gray', lw=1)
        ax.plot([dipole.an.pos.real, p2.real],[dipole.an.pos.imag, p2.imag], zorder=0, color='gray', lw=1)
        ax.set_aspect('equal')
        ax.set(xlim=(-0.5 * region_size, 0.5 * region_size), ylim=(-0.5 * region_size, 0.5 * region_size))
    plt.pause(0.001)
    anime.append(GetFrame())

size = anime[0].shape[0:2][::-1]
fps = 30
out = cv2.VideoWriter('MD_naive.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size, True)
for frame in anime:
    data = frame
    out.write(data)
out.release()
