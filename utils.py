import numpy as np
from dynamics import step, unicycle_ddt
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh
import irispy

def constrain_velocity(vel_desired, p0, A, a):
    speed_desired = np.linalg.norm(vel_desired)
    dir_desired = vel_desired / speed_desired
    r = float((p0 - a) @ A @ (p0 - a)[:,None])
    k = 5
    cbf = k * (1 - r)
    spd_cmd = min(cbf, speed_desired)
    return spd_cmd * dir_desired

def compute_pf_control(tracker, v_desired):
    speed_desired = np.linalg.norm(v_desired)
    direction_desired = v_desired / speed_desired
    cos_desired = direction_desired[0]
    sin_desired = direction_desired[1]
    pos = tracker.unicycle_state[:2]
    theta = tracker.unicycle_state[2]
    cos_now = np.cos(theta)
    sin_now = np.sin(theta)
    vel = tracker.unicycle_state[3]
    min_lin_acc = tracker.params.min_linear_acceleration
    max_lin_acc = tracker.params.max_linear_acceleration
    min_ang_acc = tracker.params.min_angular_acceleration
    max_ang_acc = tracker.params.max_angular_acceleration
    pos = tracker.unicycle_state[:2]
    theta = tracker.unicycle_state[2]
    omega = tracker.unicycle_state[4]
    direction = np.array([np.cos(theta), np.sin(theta)])
    vel = tracker.unicycle_state[3]
    min_lin_acc = tracker.params.min_linear_acceleration
    max_lin_acc = tracker.params.max_linear_acceleration
    min_ang_acc = tracker.params.min_angular_acceleration
    max_ang_acc = tracker.params.max_angular_acceleration
    

    qminusp = np.array([cos_now, sin_now]) - direction_desired
    dist = np.linalg.norm(qminusp)
    #ddtheta = np.dot(qminusp, np.array([sin_now, -cos_now]))
    ddtheta = np.dot(qminusp, np.array([-sin_now, cos_now]))
    if ddtheta > 0:
        alpha = dist * min_ang_acc
    else:
        alpha = dist * max_ang_acc

    alpha += - 2*omega * abs(omega)

    if vel < speed_desired and dist < np.sqrt(2):
        acceleration = abs(vel - speed_desired) * max_lin_acc
    else:
        acceleration = abs(vel - speed_desired) * min_lin_acc

    return np.array([acceleration, alpha])

    

def compute_viewpoint_ref(target):
    pos = target.unicycle_state[:2]
    theta = target.unicycle_state[2]

    r = 1 # eventually this should probably be a parameter somewhere?
    viewpoint_ref = pos + r * np.array([np.cos(theta), np.sin(theta)])
    return viewpoint_ref

def find_ellipse_intersection(A, B, a, b):
    lower = 0
    upper = 1
    while 1:
        if abs(lower - upper) < .00001:
            raise Exception('Invalid ellipses')
        lam = (lower + upper) / 2.
        e_lambda = lam * A + (1 - lam) * B
        m_lambda = np.linalg.inv(e_lambda) @ (lam * A @ a[:, None] + (1 - lam) * B @ b[:,None])

        dist = (a[:,None] - m_lambda).T @ A @ (a[:,None] - m_lambda)
        in_A = dist < 1
        dist = (b[:,None] - m_lambda).T @ B @ (b[:,None] - m_lambda)
        in_B = dist < 1

        if in_A and not in_B:
            #lower = lam
            upper = lam
        elif in_B and not in_A:
            #upper = lam
            lower = lam
        elif in_A and in_B:
            break
        else:
            raise Exception('Ellipses do not overlap!')

    return np.squeeze(m_lambda)

def call_irispy(env, seed):
    obs = [arr.T for arr in env.obstacles]
    bounds = irispy.Polyhedron.fromBounds(*env.bounds)

    region = irispy.inflate_region(obs, seed, bounds)
    return region


def rollout(state, params, n_steps, dt):
    positions = np.zeros((n_steps, 2))
    times = np.zeros(n_steps)
    x = state[:5]
    u = np.zeros(2)
    p = params
    for ix in range(n_steps):
        positions[ix] = x[:2]
        times[ix] = ix * dt
        x = x + unicycle_ddt(x, u, p) * dt
    return times, positions


def mvee(points, tol = 0.001):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * np.linalg.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u-u)
        u = new_u
    c = u*points
    A = np.linalg.inv(points.T*np.diag(u)*points - c.T*c)/d    
    return np.asarray(A), np.squeeze(np.asarray(c))

def check_view(tracker, targets, target_ix):
    #tracker = trackers.agent_list[0]
    target = targets.agent_list[target_ix]

    x_diff = tracker.unicycle_state[:2] - target.unicycle_state[:2]
    x_dist = np.linalg.norm(x_diff)
    x_diff_heading = x_diff / x_dist

    target_heading = np.array([np.cos(target.unicycle_state[2]), np.sin(target.unicycle_state[2])])

    cos_theta = np.dot(x_diff_heading, target_heading)
    if cos_theta > (np.sqrt(3) / 2.0) and x_dist < 2:
        return True
    else:
        return False


def update_switch_waypoint(target_traj, traj, switch_ix):
    traj_pad = np.hstack([np.zeros(5), np.array(traj).flatten()])
    traj_mat = np.reshape(traj_pad, (41, 7))
    pos = traj_mat[1:, 0:2]

    target_pos = target_traj[:,:2]
    pos_diff = pos - target_pos
    dists = np.linalg.norm(pos_diff, axis=1)
    
    new_ix = np.argmax(dists < .5)
    if new_ix == 0:
        return min(switch_ix + 3, 39)
    else:
        return new_ix + 1 # +1 is just a little buffer, not technically necessary


def update_switch_viewpoint(target_traj, traj, switch_ix):
    traj_pad = np.hstack([np.zeros(5), np.array(traj).flatten()])
    traj_mat = np.reshape(traj_pad, (41, 7))
    pos = traj_mat[1:, 0:2]

    target_pos = target_traj[:,:2]
    pos_diff = pos - target_pos
    pos_diff_heading = pos_diff / np.linalg.norm(pos_diff)

    target_heading = np.array([np.cos(target_traj[:,2]), np.sin(target_traj[:,2])]).T

    cos_theta = np.sum(pos_diff_heading * target_heading, axis=1)
    new_ix = np.argmax(cos_theta > np.sqrt(3)/2.0)
    if new_ix == 0:
        return min(switch_ix + 3, 39)
    else:
        return new_ix + 1 # +1 is just a little buffer, not technically necessary

def predict_target(target):
    prediction = np.zeros((40, 3))
    state = np.copy(target.unicycle_state)
    u = np.zeros(3)
    p = target.params

    prediction[0] = state[:3]
    for ix in range(39):
        state = step(state, u, p, 5.0/40, mpcc=True)
        prediction[ix+1] = state[:3]
    return prediction


def K_full(s, v, A_inv, B_inv):
    return 1 - v @ np.linalg.inv(A_inv / (1-s) + B_inv / s) @ v[:,None] 

def ellipsoids_intersect(A, B, a, b):
    # assumes ellipse of form (x - a)' A (x - a) < 1
    res = minimize_scalar(K_full, bounds=(0., 1.), args=(a - b, np.linalg.inv(A), np.linalg.inv(B)), method='bounded')
    return (res.fun >= 0)

# From: https://math.stackexchange.com/questions/1114879/detect-if-two-ellipses-intersect
#def K(x, dd, v):
#    return 1. - np.sum(v * ((dd * x * (1. - x)) / (x + dd * (1. - x))) * v)
#def ellipsoids_intersect(A, B, a, b):
#    # assumes ellipse of form (x - a)' A (x - a) < 1
#    dd, Phi = eigh(A, B, eigvals_only=False)
#    v = np.dot(Phi.T, a - b)
#    res = minimize_scalar(K, bounds=(0., 1.), args=(dd, v), method='bounded')
#    return (res.fun >= 0)



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

