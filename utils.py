import numpy as np
from dynamics import step, unicycle_ddt


def rollout(state, params, n_steps, dt):
    positions = np.zeros((n_steps, 2))
    times = np.zeros(n_steps)
    x = state
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
    u = np.zeros(2)
    p = target.params

    prediction[0] = state[:3]
    for ix in range(39):
        state = step(state, u, p, 5.0/40)
        prediction[ix+1] = state[:3]
    return prediction
