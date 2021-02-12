import numpy as np
from dynamics import step, unicycle_ddt
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh


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

def update_switch(targets, traj, current_target_ix, switch_ix):
    traj_pad = np.hstack([np.zeros(5), np.array(traj).flatten()])
    traj_mat = np.reshape(traj_pad, (41, 7))
    pos = traj_mat[:, 0:2]

    target_state = targets.agent_list[current_target_ix].unicycle_state
    target_pos = target_state[:2]
    pos_diff = pos - target_pos
    pos_diff_heading = pos_diff / np.linalg.norm(pos_diff)

    target_heading = np.array([np.cos(target_state[2]), np.sin(target_state[2])])

    cos_theta = np.dot(pos_diff_heading, target_heading)
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

def ellipsoids_intersect2(A, B, a, b):
    # assumes ellipse of form (x - a)' A (x - a) < 1
    res = minimize_scalar(K_full, bounds=(0., 1.), args=(a - b, np.linalg.inv(A), np.linalg.inv(B)), method='bounded')
    return (res.fun >= 0)

# From: https://math.stackexchange.com/questions/1114879/detect-if-two-ellipses-intersect
def K(x, dd, v):
    return 1. - np.sum(v * ((dd * x * (1. - x)) / (x + dd * (1. - x))) * v)
def ellipsoids_intersect(A, B, a, b):
    # assumes ellipse of form (x - a)' A (x - a) < 1
    dd, Phi = eigh(A, B, eigvals_only=False)
    v = np.dot(Phi.T, a - b)
    res = minimize_scalar(K, bounds=(0., 1.), args=(dd, v), method='bounded')
    return (res.fun >= 0)


def get_path(pred, i, j):
    path = [j]
    k = j
    while pred[i, k] != -9999:
        path.append(pred[i, k])
        k = pred[i, k]
    return path[::-1]

