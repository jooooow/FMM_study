import io
import cv2
import math
import cmath
import random
import itertools
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib.lines as lines

seed = random.randint(0, 1000)
print(f'seed = {seed}')
random.seed(seed)
atom_size = 0.05
region_size = 3
m = 1

class Atom:
    def __init__(self, q, pos, v):
        self.q = q
        self.pos = pos
        self.pre_pos = [pos, pos]
        self.potential = complex(0)
        self.potential2 = complex(0)
        self.force = complex(0)
        self.force2 = complex(0)
        self.acc = complex(0)
        self.v = v

        self.pos_pre = pos
        self.v_pre = v
        self.acc_pre = complex(0)

        self.cell_idx = complex(0)
    def __repr__(self):
        return f'atom(q={self.q}, pos={self.pos}, v={self.v})'

def random_complex():
    l = random.uniform(0,1)
    theta = random.uniform(-math.pi,math.pi)
    return complex(l * math.cos(theta), l * math.sin(theta))

def GetAtomCircle(atom):
    if atom.q > 0:
        return plt.Circle((atom.pos.real, atom.pos.imag), atom_size, color = 'r')
    elif atom.q < 0:
        return plt.Circle((atom.pos.real, atom.pos.imag), atom_size, color = 'b')
    return plt.Circle((atom.pos.real, atom.pos.imag), atom_size, fill = False)

def GetCircle(pos, size, color):
    return plt.Circle((pos.real, pos.imag), size, color = color)


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

def GetFrame():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    dst = cv2.imdecode(enc, 1)
    dst = dst[:,:,::-1]
    return dst

M = 8
offset = region_size / M / 2 + 0.3
atoms = []
for y in np.linspace(-region_size / 2 + offset, region_size / 2 - offset, M):
    for x in np.linspace(-region_size / 2 + offset, region_size / 2 - offset, M):
        atoms.append(Atom((random.randint(0, 1) * 2 - 1) * 0.5, complex(x, y) + random_complex() * 0.0, complex(0)))

N = 4
P = 3
atom_cell_dict = {}
cell_dict = {complex(x,y):[[0]*(P+1),[0]*(P+1)] for  x,y in itertools.product(range(N), range(N))}
color_dict = {complex(x,y):str('#'+str(hex(random.randint(0, pow(2,24)-1)))[2:]) for x,y in itertools.product(range(N), range(N))}

def update_dict(atom, idx):
    if atom.cell_idx not in atom_cell_dict:
            atom_cell_dict[atom.cell_idx] = []
    atom_cell_dict[atom.cell_idx].append(idx)

def get_cell_center(cell_idx):
    return (cell_idx / N + complex(1 / N / 2 - 0.5, 1 / N / 2 - 0.5)) * region_size

fig, ax = plt.subplots()
anime = []
dt = 0.005
duration = 50
rms_p = []
rms_f = []
for i in range(duration):
    time = dt * i
    plt.cla()

    # update cell idx & clear potential/force
    atom_cell_dict = {}
    for idx,a_tar in enumerate(atoms):
        a_tar.cell_idx = (a_tar.pos / region_size + complex(0.5, 0.5)) * N
        a_tar.cell_idx = complex(int(a_tar.cell_idx.real), int(a_tar.cell_idx.imag))
        update_dict(a_tar, idx)
        a_tar.potential2 = 0
        a_tar.force2 = complex(0)

    # P2M
    for cell_idx, atom_idx_list in atom_cell_dict.items():
        for k in range(0,P+1):
            cell_dict[cell_idx][0][k] = complex(0)
            for atom_idx in atom_idx_list:
                a = atoms[atom_idx]
                if k >= 1:
                    cell_dict[cell_idx][0][k] += -1 / k * a.q * (a.pos - get_cell_center(cell_idx)) ** k
                else:
                    cell_dict[cell_idx][0][k] += a.q

    # M2L & P2P
    for cell_idx_tar in atom_cell_dict:
        for t in range(0, P+1):
            cell_dict[cell_idx_tar][1][t] = complex(0)
        for cell_idx_src in atom_cell_dict:
            r = cell_idx_src - cell_idx_tar
            r_x, r_y = abs(r.real), abs(r.imag)
            '''if r_x > N / 2:
                r_x = N - r_x
            if r_y > N / 2:
                r_y = N - r_y'''
            r = complex(r_x, r_y)
            # P2P
            if r.real <= 1 and r.imag <= 1:
                for atom_tar_idx in atom_cell_dict[cell_idx_tar]:
                    ptemp1 = complex(0)
                    ftemp1 = complex(0)
                    for atom_src_idx in atom_cell_dict[cell_idx_src]:
                        if atom_tar_idx == atom_src_idx:
                            continue
                        r = atoms[atom_tar_idx].pos - atoms[atom_src_idx].pos
                        r = warp(r)
                        ptemp1 += -atoms[atom_src_idx].q * cmath.log(r)
                        ftemp1 += atoms[atom_tar_idx].q * atoms[atom_src_idx].q * 1 / (r)
                    atoms[atom_tar_idx].potential2 += ptemp1.real
                    atoms[atom_tar_idx].force2 += ftemp1
            # M2L
            else:
                pos_tar = get_cell_center(cell_idx_tar)
                pos_src = get_cell_center(cell_idx_src)
                for t in range(0, P+1):
                    zdiv = pos_tar - pos_src
                    zdiv = warp(zdiv)
                    if t == 0:
                        cell_dict[cell_idx_tar][1][0] += cell_dict[cell_idx_src][0][0] * cmath.log(zdiv) + sum(cell_dict[cell_idx_src][0][l] / zdiv ** l for l in range(1,P+1))
                    else:
                        cell_dict[cell_idx_tar][1][t] += -cell_dict[cell_idx_src][0][0] / (t * (-zdiv) ** t) + 1 / ((-zdiv) ** t) * sum(cell_dict[cell_idx_src][0][l] / ((-zdiv) ** l) * comb(t + l - 1, l - 1) * (-1) ** l for l in range(1,P+1))
            
    # L2P
    for cell_idx, atom_idx_list in atom_cell_dict.items():
        cell_pos = get_cell_center(cell_idx)
        for atom_idx in atom_idx_list:
            for t in range(0, P+1):
                atoms[atom_idx].potential2 += -((atoms[atom_idx].pos - cell_pos) ** t * cell_dict[cell_idx][1][t]).real
                atoms[atom_idx].force2 += atoms[atom_idx].q * t * ((atoms[atom_idx].pos - cell_pos) ** (t - 1)) * cell_dict[cell_idx][1][t]
            atoms[atom_idx].force2 = complex(atoms[atom_idx].force2.real, -atoms[atom_idx].force2.imag)
                
    # direct potential & force
    for a_tar in atoms:
        potential = complex(0)
        ptemp = 0
        force = complex(0)
        ftemp = complex(0)
        for a_src in atoms:
            if a_tar != a_src:
                r = a_tar.pos - a_src.pos
                r = warp(r)
                ptemp += -math.log(abs(r)) * a_src.q
                phi = -a_src.q * cmath.log(r)
                f = a_tar.q * a_src.q * 1 / (r)
                ftemp += a_tar.q *  a_src.q * 1 / abs(r) * r / abs(r)
                potential += phi
                force += f
        #a_tar.force = complex(force.real, -force.imag)
        #a_tar.potential = potential.real
        a_tar.force = ftemp
        a_tar.potential = ptemp#

    # update acc, v, pos
    p1 = []
    p2 = []
    fs = []
    ps = []
    for a_tar in atoms:
        #1.normal 
        '''a_tar.acc = a_tar.force2 / m
        a_tar.v += a_tar.acc * dt
        a_tar.pos += a_tar.v * dt'''
        #2.verlet
        '''a_tar.acc = a_tar.force2 / m
        a_tar.pos = 2 * a_tar.pre_pos[0] - a_tar.pre_pos[1] + a_tar.acc * dt * dt
        a_tar.pre_pos[1] = a_tar.pre_pos[0]
        a_tar.pre_pos[0] = a_tar.pos'''
        #3.velocity-verlet
        a_tar.pos = a_tar.pos_pre + a_tar.v_pre * dt + 0.5 * a_tar.acc_pre * dt * dt
        a_tar.pos = warp(a_tar.pos)
        a_tar.acc = a_tar.force2 / m
        a_tar.v = a_tar.v_pre + 0.5 * (a_tar.acc_pre + a_tar.acc) * dt
        a_tar.pos_pre = a_tar.pos
        a_tar.acc_pre = a_tar.acc
        a_tar.v_pre = a_tar.v

        #print(pre_pos)

        p1.append(a_tar.potential)
        p2.append(a_tar.potential2)
        fs.append(abs(a_tar.force - a_tar.force2) ** 2)
        ps.append(abs(a_tar.potential - a_tar.potential2) ** 2)

    pss = math.sqrt(sum(ps) / len(ps))
    fss = math.sqrt(sum(fs) / len(fs))
    rms_p.append(pss)
    rms_f.append(fss)

    print('t = {:.3f}({:.2f}%), (ps,fs) = ({:.6f},{:.6f})'.format(
        time,
        i / duration * 100,
        pss, 
        fss)
    )
    
    # visualization
    for atom in atoms:
        ax.add_artist(GetAtomCircle(atom))
        ax.set_title('t={:.3f}s'.format(time), loc='left')
        ax.set_aspect('equal')
        ax.set(xlim=(-0.5 * region_size, 0.5 * region_size), ylim=(-0.5 * region_size, 0.5 * region_size))
    #plt.pause(0.001)
    anime.append(GetFrame())

size = anime[0].shape[0:2][::-1]
fps = 30
out = cv2.VideoWriter('md_FMM_2d_deep1_velocity-verlet_warp.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size, True)
for frame in anime:
    data = frame
    out.write(data)
out.release()

plt.figure()
plt.plot(rms_p)
plt.ylim([-0,1])
plt.show()