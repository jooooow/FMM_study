# simulation of intermolecule interaction
# using LJ potential

from mpl_toolkits import mplot3d
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

epsilon = 1
sigma = 0.2
rc = 3 * sigma
m = 1
radius = 0.1
size = 10

class molecule:
    def __init__(self, pos):
        self.pos = pos
        self.v = complex(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        self.f = complex(0)
    def __repr__(self):
        return f'molecule({self.pos}, {self.v})'

def LJ_force(r, epsilon, sigma):
    return 24 * epsilon * (2 * (sigma ** 12) / (r ** 13) - (sigma ** 6) / (r ** 7))

def truncated_LT_force(r, rc, epsilon, sigma):
    if r <= rc:
        return LJ_force(r, epsilon, sigma) - LJ_force(rc, epsilon, sigma)
    return 0

def warp(x):
    r = x
    if r.real > 0.5 * size:
        r -= complex(size,0)
    elif r.real < -0.5 * size:
        r += complex(size,0)
    if r.imag > 0.5 * size:
        r -= complex(0,size)
    elif r.imag < -0.5 * size:
        r += complex(0,size)
    return r

ms = []
N = 25
for x in np.linspace(-4.5, 4.5, N):
    for y in np.linspace(-4.5, 4.5, N):
        ms.append(molecule(complex(x, y)))

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set(xlim=(0, 3), ylim=(0, 3))
theta = np.linspace(0, 2*np.pi, 100)
anime = []
dt = 0.005

for i in range(0, 1000):
    print(f't = {dt*i}')
    plt.cla()
    for mtar in ms:
        force = complex(0)
        for msrc in ms:
            if msrc != mtar:
                r = mtar.pos - msrc.pos
                r = warp(r)
                f = truncated_LT_force(abs(r), rc, epsilon, sigma)
                direction = r / abs(r)
                f = f * direction
                force = force + f
        mtar.f = force

    for mtar in ms:
        acc = mtar.f / m
        mtar.v += acc * dt
        mtar.pos += mtar.v * dt
        mtar.pos = warp(mtar.pos)

        #print(r, abs(r), direction, f, force)
            
        x = radius*np.cos(theta) + mtar.pos.real
        y = radius*np.sin(theta) + mtar.pos.imag
        plt.plot(x,y)

    ax.set_aspect('equal')
    ax.set(xlim=(-0.5 * size, 0.5 * size), ylim=(-0.5 * size, 0.5 * size))
    
    plt.pause(0.001)
    fig = ax.figure
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    anime.append(data)

size = anime[0].shape[0:2][::-1]
fps = 30
out = cv2.VideoWriter('LT2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size, True)

cnt = 0
for frame in anime:
    if cnt % 2 == 0:
        data = frame
        out.write(data)
    cnt = cnt + 1

out.release()

