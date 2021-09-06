import numpy as np
import random
from shapely import geometry
from utils import call_irispy, ellipsoids_intersect, find_ellipse_intersection
from scipy.sparse.csgraph import shortest_path
from environment import Environment

def sample_tracker_positions(M_list, center_list, n_trackers, sample_bound):
    positions = []
    indices = list(range(len(M_list)))
    while len(positions) < n_trackers:
        p = np.random.uniform(-sample_bound, sample_bound, size=2)
    
        random.shuffle(indices)
        for ix in indices:
            M = M_list[ix]
            c = center_list[ix]
            r = (c - p) @ M @ (c - p)[:,None]
            if r < 1:
                positions.append(p)
                break

    return positions

# 1) spawn obstacles
def construct_environment(n_obs, bound):
    """ Makes an environment of random triangles """
    obstacles = []
    for ix in range(n_obs):
        verts = 5 * (np.random.rand(3,2) - 0.5)
        verts += bound*(np.random.rand(1,2) - 0.5)
        obstacles.append(verts)
    env = Environment(obstacles, [[-bound, -bound], [bound, bound]])
    return env

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
            

# 2) Ellipse space construction
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
                graph[ix, jx] = 1
                graph[jx, ix] = 1


    return graph

def construct_ellipse_topology_weighted(M_list, center_list):
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



def find_ellipses_for_point(M_list, center_list, point):
    #indices = []
    distances = np.zeros(len(M_list))
    for ix in range(len(M_list)):
        dist = (point - center_list[ix]) @ M_list[ix] @ (point - center_list[ix])[:,None]
        distances[ix] = dist
        #if dist < 1.:
        #    indices.append(ix)
    indices = np.argwhere(distances < 1.)
    if len(indices) > 0:
        return np.squeeze(indices)
    else:
        return np.argmin(distances)

    #return indices


def find_ellipsoid_path(graph, M_list, center_list, start, end):

    start_ix_list = find_ellipses_for_point(M_list, center_list, start)
    end_ix_list = find_ellipses_for_point(M_list, center_list, end)


    # We augment the ellipse connectivity graph with nodes for the start and end position
    # The connectivity of these 2 nodes is determined by which ellipses they are contained in
    n = graph.shape[0]
    graph_aug = np.zeros((n+2, n+2))
    graph_aug[:n, :n] = graph
    graph_aug[n, start_ix_list] = 1
    graph_aug[start_ix_list, n] = 1
    graph_aug[n + 1, end_ix_list] = 1
    graph_aug[end_ix_list, n + 1] = 1
    

    D, pred = shortest_path(graph_aug, directed=False, method='FW', return_predecessors=True)
    ellipse_path = get_path(pred, n, n+1)
    ellipse_path = ellipse_path[1:-1]
    ellipse_pairs = [(ellipse_path[ix], ellipse_path[ix+1]) for ix in range(len(ellipse_path) - 1)]
    print(ellipse_pairs)
    waypoints = np.array([find_ellipse_intersection(M_list[a], M_list[b], center_list[a], center_list[b]) for (a,b) in ellipse_pairs])
    shape_matrices = [M_list[ix] for ix in ellipse_path]
    offsets = [center_list[ix] for ix in ellipse_path]

    return ellipse_path, waypoints, shape_matrices, offsets

def find_ellsoid_path_weighted(graph, M_list, center_list, start, end):
    start_ix_list = find_ellipses_for_point(M_list, center_list, start)
    end_ix_list = find_ellipses_for_point(M_list, center_list, end)

    start_dists = [ (start - center_list[ix]) @ M_list[ix] @ (start - center_list[ix])[:, None] for ix in start_ix_list]

    end_dists = [ (end - center_list[ix]) @ M_list[ix] @ (end - center_list[ix])[:, None] for ix in end_ix_list]


    # We augment the ellipse connectivity graph with nodes for the start and end position
    # The connectivity of these 2 nodes is determined by which ellipses they are contained in
    n = graph.shape[0]
    graph_aug = np.zeros((n+2, n+2))
    graph_aug[:n, :n] = graph
    graph_aug[n, start_ix_list] = start_dists
    graph_aug[start_ix_list, n] = start_dists
    graph_aug[n + 1, end_ix_list] = end_dists
    graph_aug[end_ix_list, n + 1] = end_dists


    D, pred = shortest_path(graph_aug, directed=False, method='FW', return_predecessors=True)
    shortest_distance = D[n, n+1]
    ellipse_path = get_path(pred, n, n+1)
    ellipse_path = ellipse_path[1:-1]
    ellipse_pairs = [(ellipse_path[ix], ellipse_path[ix+1]) for ix in range(len(ellipse_path) - 1)]
    print(ellipse_pairs)
    waypoints = np.array([find_ellipse_intersection(M_list[a], M_list[b], center_list[a], center_list[b]) for (a,b) in ellipse_pairs])
    shape_matrices = [M_list[ix] for ix in ellipse_path]
    offsets = [center_list[ix] for ix in ellipse_path]

    return ellipse_path, waypoints, shape_matrices, offsets, shortest_distance



def get_path(pred, i, j):
    path = [j]
    k = j
    while pred[i, k] != -9999:
        path.append(pred[i, k])
        k = pred[i, k]
    return path[::-1]

