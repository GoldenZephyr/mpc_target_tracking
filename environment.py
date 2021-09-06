import numpy as np
from shapely import geometry

class Environment:
    def __init__(self, obstacles, bounds):
        """
        obstacles is a list of Nx2 numpy arrays
        """
        self.obstacles = obstacles
        self.bounds = bounds
        self.shapely_obstacles = [geometry.Polygon(o) for o in obstacles]


        #self.centers = np.array([np.mean(verts, axis=0) for verts in obstacles])
        self.centers = []
        self.perpendiculars = []
        self.side_perpendiculars = []
        self.sides = []

        for o in obstacles:
            for ix in [ [0,1], [1,2], [2,0] ]:
                center = np.mean(o, axis=0)
                side = o[ix]
                P, side_perp = construct_obstacle_edge_field(side, center)
                self.perpendiculars.append(P)
                self.side_perpendiculars.append(side_perp)
                self.centers.append(center)
                self.sides.append(side)

def construct_obstacle_edge_field(verts, center):
    line1 = verts[1] - verts[0]
    perp_candidate = np.array([-line1[1], line1[0]])
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

    return P, A_perp

def calculate_net_obstacle_forces(env, Pt, Ptdir):
    force = 0
    for ix in range(len(env.centers)):
        Apt = env.sides[ix][0]
        Bpt = env.sides[ix][1]
        Pdir = env.perpendiculars[ix]
        Adir = env.side_perpendiculars[ix]
        force += calculate_obstacle_forces(Pt, Ptdir, Apt, Bpt, Pdir, Adir)
    return force

def calculate_obstacle_forces(Pt, Ptdir, Apt, Bpt, Pdir, Adir):
    a = Pt - Apt
    if np.dot(a, Adir) < 0:
        return 0
    b = Pt - Bpt
    if np.dot(b, -Adir) < 0:
        return 0
    p = Pt - Apt
    dist = np.dot(p, Pdir)
    if dist < 0:
        return 0
    else:
        sin_theta = float(np.cross(-Pdir, Ptdir))
        sgn = np.sign(sin_theta)
        force = 10 * sgn * (1 - abs(sin_theta)) / (dist**2)
        return force

