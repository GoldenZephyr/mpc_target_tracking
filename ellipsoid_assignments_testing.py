#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from environment import Environment
from scipy.sparse.csgraph import shortest_path
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull
from shapely import geometry
import irispy
from visualization import EnvironmentPlotCxt


def K_full(s, v, A_inv, B_inv):
    return 1 - v @ np.linalg.inv(A_inv / (1-s) + B_inv / s) @ v[:,None]


def ellipsoids_intersect(A, B, a, b):
    # assumes ellipse of form (x - a)' A (x - a) < 1
    res = minimize_scalar(K_full, bounds=(0., 1.), args=(a - b, np.linalg.inv(A), np.linalg.inv(B)), method='bounded')
    return (res.fun >= 0)


def call_irispy(env, seed):
    obs = [arr.T for arr in env.obstacles]
    bounds = irispy.Polyhedron.fromBounds(*env.bounds)

    region = irispy.inflate_region(obs, seed, bounds)
    return region


def construct_environment_blocks(bound):
    block_base = 3 * np.array([[0,0], [0,1], [1,1], [1,0]])

    obstacles = []

    offsets = [-9, -2, 5, 12]
    for o1 in offsets:
        xoff = np.zeros((4,2))
        xoff[:,0] = o1
        for o2 in offsets:
            yoff = np.zeros((4,2))
            yoff[:,1] = o2
            obstacles.append(block_base + xoff + yoff)

    env = Environment(obstacles, [[-bound, -bound], [bound, bound]])
    return env

def construct_environment_forest(bound):


    # we do this so that the randomness to generate this environment
    # is decoupled from other randomness
    rng_state = np.random.get_state()
    np.random.seed(2)
    thetas = np.linspace(0,2*np.pi, 12)
    unit_circle = np.array([np.cos(thetas), np.sin(thetas)]).T
    obstacles = []
    for ix in range(20):
        center = 20 * (np.random.random(2) - 0.5)
        radius = 3 * np.random.random()
        obs = radius*unit_circle + center
        obstacles.append(obs)

    np.random.set_state(rng_state)

    env = Environment(obstacles, [[-bound, -bound], [bound, bound]])
    return env

def construct_ellipse_space(env):

    shapely_obstacles = [geometry.Polygon(o) for o in env.obstacles]
    min_bounds, max_bounds = env.bounds
    x = np.linspace(min_bounds[0] + 1, max_bounds[0] - 1, 30)
    y = np.linspace(min_bounds[1] + 1, max_bounds[1] - 1, 30)

    #x = np.linspace(-8, 8, 30)
    #y = np.linspace(-8, 8, 30)

    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()
    yv = yv.ravel()
    needed_points = [True] * len(xv)
    vals = np.vstack([xv, yv]).T
    shapely_points = [geometry.Point(p) for p in vals]
    ix_list = np.arange(len(xv))
    C_list = []
    M_list = []
    center_list = []
    region_list = []

    while(any(needed_points)):
        first_ix = np.argmax(needed_points)
        needed_points[first_ix] = False
        point_in_obstacle = [o.contains(shapely_points[first_ix]) for o in shapely_obstacles]
        if any(point_in_obstacle):
            continue
        seed_point = vals[first_ix]
        region = call_irispy(env, seed_point)
        d = region.ellipsoid.getD()
        c = region.ellipsoid.getC()
        c_inv = np.linalg.inv(c)
        c_inv_sq = c_inv @ c_inv
        M_list.append(c_inv_sq)
        region_list.append(region)
        C_list.append(c) # shape matrix
        center_list.append(d) # center
        ix_remaining = ix_list[needed_points]
        for ix in ix_remaining:
            dist = (vals[ix] - d) @ c_inv_sq @ (vals[ix] - d)[:,None]
            eps = .01 # need this because sometimes the ellipsoid fits the seed point *on* the ellipse boundary
            if dist < 1 + eps:
                needed_points[ix] = False

    return region_list, M_list, C_list, center_list


def construct_ellipse_topology(M_list, center_list):
    n = len(M_list)
    graph = np.zeros((n,n))
    for ix in range(n):
        A = M_list[ix]
        a = center_list[ix]
        for jx in range(ix + 1, n):
            B = M_list[jx]
            b = center_list[jx]
            if ellipsoids_intersect(A, B, a, b):

                lams = np.linspace(0, 1, 20)
                e_lams = [l*A + (1-l) * B for l in lams]
                m_lams = np.array([np.linalg.inv(el) @ (l * A @ a[:, None] + (1 - l) * B @ b[:,None]) for (el, l) in zip(e_lams, lams)]).squeeze()

                diffs = np.diff(m_lams, axis=0)
                dists = np.linalg.norm(diffs, axis=1)
                total_distance = np.sum(dists)

                #graph[ix, jx] = 1
                #graph[jx, ix] = 1

                graph[ix, jx] = total_distance
                graph[jx, ix] = total_distance
            else:
                graph[ix, jx] = np.inf
                graph[jx, ix] = np.inf

    return graph


def greedy_center_selection(g, k):
    n = len(g)
    centers = [np.random.randint(0, n)]

    while len(centers) < k:
        center_vector = np.array(centers)
        dists = g[center_vector,:] 
        print(dists)
        #dists[:,np.array(centers)] = 0
        maxdists = np.min(dists, axis=0)
        #if len(centers) > 1:
        #    maxdists = np.min(dists, axis=0)
        #else:
        #    maxdists = dists.squeeze()
        print(maxdists)
        #dists[:,np.array(centers)] = 0
        new_center = np.argmax(maxdists)
        print(new_center)
        centers.append(new_center)

    center_vector = np.array(centers)
    dists = g[center_vector,:]
    assignments = np.argmin(dists, axis=0)
    return centers, assignments



#env =construct_environment_blocks(15)
env =construct_environment_forest(15)

region_list, M_list, C_list, center_list = construct_ellipse_space(env)
ellipse_graph = construct_ellipse_topology(M_list, center_list)
plt.imshow(ellipse_graph)
plt.show()

fig, ax = plt.subplots()
env_cxt = EnvironmentPlotCxt(ax, env, C_list, center_list)
plt.show()

D, pred = shortest_path(ellipse_graph, directed=False, method='FW', return_predecessors=True)

eg2 = np.copy(ellipse_graph)
#for ix in range(len(D)):
#    for jx in range(ix+1, len(D)):
#        if ellipse_graph[ix,jx] < np.inf and D[ix, jx] < ellipse_graph[ix, jx]:
#            eg2[ix, jx] = D[ix, jx]
#            eg2[jx, ix] = D[ix, jx]


centers, assignments = greedy_center_selection(D, 4)

fig, ax = plt.subplots()
t = np.linspace(0, 2*np.pi + .1, 25)
x = np.array([np.cos(t), np.sin(t)])


patches = []
colors = ['r', 'g', 'b', 'c']
for ix in range(len(C_list)):
    C = C_list[ix]
    d = center_list[ix]
    y = C @ x + d[:, None]
    polygon = Polygon(y.T, False, ec=None, fc=colors[assignments[ix]], alpha=0.2)
    #ax.add_patch(polygon)

    poly_pts = region_list[ix].getPolyhedron().getDrawingVertices()
    hull = ConvexHull(poly_pts)
    poly2 = Polygon(poly_pts[hull.vertices], True, fc=colors[assignments[ix]], fill=True, alpha=0.2)
    ax.add_patch(poly2)
    #patches.append(polygon)
    xv = y[0, :]
    yv = y[1, :]
    #l = plt.plot(xv, yv, color=colors[assignments[ix]])

#ax.add_collection(PatchCollection(patches))
for c in centers:
    plt.scatter(center_list[c][0], center_list[c][1])

EnvironmentPlotCxt(ax, env, [], [])
plt.show()



