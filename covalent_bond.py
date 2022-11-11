# simulation of covalent bond between a dipole
# use harmonic potential v = 1/2 * k * (r - r0) ** 2

from mpl_toolkits import mplot3d
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

k = 10000
r0 = 0.1
radius = 0.03
m = 1

def distance(a1, a2):
    return math.sqrt((a1.x - a2.x) ** 2 + (a1.y - a2.y) ** 2)

class atom:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
    def __repr__(self):
        return f'atom({self.x}, {self.y})'

def covalent_force(k, r, r0):
    return -k * (r - r0)

a1 = atom(0.4, 0.4)
a2 = atom(0.5, 0.5)
print(a1)
print(a2)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set(xlim=(0, 3), ylim=(0, 3))
theta = np.linspace(0, 2*np.pi, 100)
anime = []
dt = 0.002

rs = []
for i in range(0, 1000):
    plt.cla()
    r = distance(a1, a2)
    rs.append(r)
    #print(r)
    f1 = covalent_force(k, r, r0)
    f2 = covalent_force(k, r, r0)
    acc1 = f1 / m
    acc2 = f2 / m
    direction1 = [a1.x - a2.x, a1.y - a2.y]
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = -direction1
    acc1 = acc1 * direction1
    acc2 = acc2 * direction2
    a1.vx = a1.vx + acc1[0] * dt
    a1.vy = a1.vy + acc1[1] * dt
    a2.vx = a2.vx + acc2[0] * dt
    a2.vy = a2.vy + acc2[1] * dt
    a1.x = a1.x + a1.vx * dt
    a1.y = a1.y + a1.vy * dt
    a2.x = a2.x + a2.vx * dt
    a2.y = a2.y + a2.vy * dt
    
    x1 = radius*np.cos(theta) + a1.x
    y1 = radius*np.sin(theta) + a1.y

    x2 = radius*np.cos(theta) + a2.x
    y2 = radius*np.sin(theta) + a2.y

    ax.set_aspect('equal')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.pause(0.001)
    fig = ax.figure
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    anime.append(data)

size = anime[0].shape[0:2][::-1]
fps = 30
out = cv2.VideoWriter('dipole.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size, True)

cnt = 0
for frame in anime:
    if cnt % 2 == 0:
        data = frame
        out.write(data)
    cnt = cnt + 1

out.release()

ax.set_aspect('auto')
plt.cla()
plt.plot(rs)
plt.show()


