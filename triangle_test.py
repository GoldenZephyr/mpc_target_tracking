#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

verts = 5 * (np.random.rand(3,2) - 0.5)
verts += 5*(np.random.rand(1,2) - 0.5)

center = np.mean(verts, axis=0)
plt.scatter([center[0]], [center[1]], color='k')
mp1 = (verts[0] + verts[1]) / 2.
mp2 = (verts[1] + verts[2]) / 2.
mp3 = (verts[2] + verts[0]) / 2.

mp_center = mp1 - center
#perp_candidate = np.array([-mp1[1], mp1[0]])
line1 = verts[1] - verts[0]
perp_candidate = np.array([-line1[1], line1[0]])
#if np.dot(mp_center, perp_candidate) > 0:
if np.dot(verts[0], perp_candidate) > 0:
    P = perp_candidate
else:
    P = -perp_candidate

Apt = verts[0]
A_dir = verts[1] - verts[0]
A_dir = A_dir / np.linalg.norm(A_dir)
A_to_center = center - Apt
if np.dot(A_dir, A_to_center) > 0:
    A_perp = A_dir
else:
    A_perp = -A_dir


Bpt = verts[1]
B_dir = verts[1] - verts[0]
B_dir = B_dir / np.linalg.norm(B_dir)
B_to_center = center - Bpt
if np.dot(B_dir, B_to_center) > 0:
    B_perp = B_dir
else:
    B_perp = -B_dir

def calculate_obstacle_forces(Pt, Ptdir, Ppt, Pdir, Apt, Adir, Bpt, Bdir):
    a = Pt - Apt
    if np.dot(a, Adir) < 0:
        return 0
    b = Pt - Bpt
    if np.dot(b, Bdir) < 0:
        return 0
    p = Pt - Ppt
    dist = np.dot(p, Pdir)
    if dist < 0:
        return 0
    else:
        sin_theta = np.cross(-Pdir, Ptdir)[0]
        sgn = np.sign(sin_theta)
        force = sgn * (1 - abs(sin_theta)) / (dist**2)
        return force


def dist_in_box(pt, Ppt, Pdir, Apt, Adir, Bpt, Bdir):
    a = pt - Apt
    if np.dot(a, Adir) < 0:
        return 0
    b = pt - Bpt
    if np.dot(b, Bdir) < 0:
        return 0
    p = pt - Ppt
    dist = np.dot(p, Pdir)
    if dist < 0:
        return 0
    else:
        return dist

plt.plot(verts[[0,1,2,0], 0], verts[[0,1,2,0], 1])
x = np.linspace(-8, 8, 30)
y = np.linspace(-8, 8, 30)

xv, yv = np.meshgrid(x, y)
xv = xv.ravel()
yv = yv.ravel()

pts = [np.array([xv[i], yv[i]]) for i in range(len(xv))]

#dists = [dist_in_box(pt, mp1, P, Apt, A_perp, Bpt, B_perp) for pt in pts]
dists = [dist_in_box(pt, Apt, P, Apt, A_perp, Bpt, -A_perp) for pt in pts]

plt.scatter(xv, yv, c=dists)
plt.plot(verts[[0,1],0], verts[[0,1],1], color='k')

A_ep = verts[0] + P * 4
A_mp = (verts[0] + A_ep) / 2.
A_mp_ext = A_mp + .5 * A_perp
plt.plot([verts[0,0], A_ep[0]], [verts[0,1], A_ep[1]], color='orange')
plt.plot([A_mp[0], A_mp_ext[0]], [A_mp[1], A_mp_ext[1]], color='orange')

B_ep = verts[1] + P * 4
B_mp = (verts[1] + B_ep) / 2.
B_mp_ext = B_mp + .5 * B_perp
plt.plot([verts[1,0], B_ep[0]], [verts[1,1], B_ep[1]], color='orange')
plt.plot([B_mp[0], B_mp_ext[0]], [B_mp[1], B_mp_ext[1]], color='orange')

plt.show()








