#!/usr/bin/python3

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import random

import numpy as np

from shapely import geometry
import shapely
import casadi as cd
from utils import check_view, update_switch_viewpoint, update_switch_waypoint, predict_target, rollout, compute_viewpoint_ref, greedy_center_selection

from agents import AgentGroup, DefaultTargetParams, DefaultTrackerParams
from visualization import initial_plot_target_group, update_plot_target_group, initial_plot_tracker_group, update_plot_tracker_group, cxt_to_artists, plot_environment, EnvironmentPlotCxt, update_plot_environment

from dynamics import update_agents, step, update_targets

from mpc_obs_functions import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_targets', type=int, required=True, help='Number of target agents')
parser.add_argument('--n_trackers', type=int, required=True, help='Number of tracker agents')
parser.add_argument('--hlp_type', type=str, required=True, help='Type of high level planning to use. Choose from {ellipsoids, no_decomposition, no_decomposition_obsaware}')
parser.add_argument('--env_type', type=str, required=True, help='Environment type. Choose from {blocks, forest}')
parser.add_argument('--n_steps', type=int, required=True, help='Number of simulation steps')

parser.add_argument('--plot_mode', type=str, help='Plotting mode {show, save, none}')
parser.add_argument('--animation_name', type=str, help='Name of saved animation (only useful if plot_mode is save')

args = parser.parse_args()

n_targets = args.n_targets
n_trackers = args.n_trackers
#assignment_type = 'MobileVoronoi'
assignment_type = None # deprecated
#controller_type = 'MPC'
#HLP_TYPE = 'no_decomposition'
#HLP_TYPE = 'ellipsoids'
HLP_TYPE = args.hlp_type
if HLP_TYPE not in ['ellipsoids', 'no_decomposition', 'no_decomposition_obsaware']:
    raise Exception('Error, invalid HLP_TYPE')
#env_type = 'blocks'
env_type = args.env_type


solver_comp = cd.nlpsol('solver', 'ipopt', './nlp_iris_2.so', {'print_time':0, 'ipopt.print_level' : 0, 'ipopt.max_cpu_time': 0.6})
#solver_comp = cd.nlpsol('solver', 'ipopt', './nlp_iris_2.so', {'ipopt.max_cpu_time': 0.2})

#targets = TargetGroup(n_targets, [-5,-5,-5], [5,5,5])
targets = AgentGroup(n_targets, [-15,-15,-15], [15,15,15], DefaultTargetParams())
for t in targets.agent_list:
    t.pd_error_last = 0
    t.waypoints = []
trackers = AgentGroup(n_trackers, [-5,-5,-5], [5,5,5], DefaultTrackerParams())
#trackers.agent_list[0].unicycle_state[0] = -4
trackers.synchronize_state()

if env_type == 'blocks':
    env = construct_environment_blocks(15)
elif env_type == 'forest':
    env = construct_environment_forest(15)

region_list, M_list, C_list, center_list = construct_ellipse_space(env)
ellipse_graph = construct_ellipse_topology(M_list, center_list)

ellipse_graph_weighted = construct_ellipse_topology_weighted(M_list, center_list)
D, pred = shortest_path(ellipse_graph_weighted, directed=False, method='FW', return_predecessors=True)
decomposition_centers_ix, decomposition_assignments = greedy_center_selection(D, n_trackers)
ellipsoids_ix_per_tracker = [np.atleast_1d(np.squeeze(np.argwhere(decomposition_assignments == ix))) for ix in np.arange(n_trackers)]
print(ellipsoids_ix_per_tracker)
ellipsoids_per_tracker = [[(M_list[ix], center_list[ix]) for ix in ellipsoids_ix_per_tracker[jx]] for jx in np.arange(n_trackers)]
for (ix, t) in enumerate(trackers.agent_list):
    decomposition_center = center_list[decomposition_centers_ix[ix]]
    print(decomposition_center)
    t.unicycle_state[:2] = decomposition_center
trackers.synchronize_state()

if not args.plot_mode == 'none':
    fig, ax = plt.subplots()
    scats = initial_plot_target_group(ax, targets)
    scats_tracker = initial_plot_tracker_group(ax, trackers)
    env_cxt = EnvironmentPlotCxt(ax, env, C_list, center_list)

    # Turn on plotting of assigned regions
    t = np.linspace(0, 2*np.pi + .1, 50)
    x = np.array([np.cos(t), np.sin(t)])
    colors = ['r','g','b','c']
    for ix in range(len(C_list)):
        C = C_list[ix]
        d = center_list[ix]
        y = C @ x + d[:, None]
        #poly = Polygon(y.T, False, ec=None, fc=colors[decomposition_assignments[ix]], alpha=0.2)
        poly = Polygon(y.T, False, ec='b', fc='none', alpha=0.2)
        ax.add_patch(poly)


current_target_ix = 0
switch_ix = 39


w0 = cd.vertcat(np.random.random(282))
bounds = cd.inf
#lbw = np.array([-bounds, -bounds, -cd.inf, 0, -cd.pi/4.0, -1, -2])
#ubw = np.array([bounds, bounds, cd.inf, 3, cd.pi/4.0, 1, 2])
lbw = np.array([-bounds, -bounds, -cd.inf, 0, -3, -1, -2])
ubw = np.array([bounds, bounds, cd.inf, 3, 3, 1, 2])

lbw = np.tile(lbw, (41, 1)).flatten()[5:]
ubw = np.tile(ubw, (41, 1)).flatten()[5:]

lbg_base = np.array([0.,0,0,0,0,0,0])
ubg_base = np.array([0.,0,0,0,0,0,0])

lbg = np.tile(lbg_base, (40, 1))
ubg = np.tile(ubg_base, (40, 1))


mpc_guesses = [w0] * n_trackers


assignment_ix = [0]*n_trackers
current_target_indices = [0]*n_trackers
#trajectories = [0] * n_trackers
trajectories = [[] for _ in range(n_trackers)]
last_assignment = [None] * n_trackers

def check_visited_view(tracker, target):
    #tracker = trackers.agent_list[0]

    x_diff = tracker.unicycle_state[:2] - target.unicycle_state[:2]
    x_dist = np.linalg.norm(x_diff)
    x_diff_heading = x_diff / x_dist

    target_heading = np.array([np.cos(target.unicycle_state[2]), np.sin(target.unicycle_state[2])])

    cos_theta = np.dot(x_diff_heading, target_heading)
    if cos_theta > (np.sqrt(3) / 2.0) and x_dist < 2:
        return True
    else:
        return False




def resample_target_goal(initial_position, env):
    global M_list, center_list, ellipse_graph_weighted

    while 1:
        rand_pt = np.random.uniform(low=-15, high=15, size=(2,))
        shapely_pt = geometry.Point(rand_pt)
        point_in_obstacle = [o.contains(shapely_pt) for o in env.shapely_obstacles]
        if not any(point_in_obstacle):
            #_, waypoints, _, _ = find_ellipsoid_path(ellipse_graph, M_list, center_list, initial_position, rand_pt)
            _, waypoints, _, _,_ = find_ellipsoid_path_weighted(ellipse_graph_weighted, M_list, center_list, initial_position, rand_pt)
            if len(waypoints) > 0:
                waypoints = np.vstack((waypoints, rand_pt))
            else:
                waypoints = np.array([rand_pt])
            break

    return waypoints


def pd_control_update(agent, target_pt):
    pos = agent.unicycle_state[:2]
    yaw = agent.unicycle_state[2]
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    res = target_pt - pos
    res = res / np.linalg.norm(res)
    
    det = heading[0] * res[1] - heading[1]*res[0]
    yaw_control = det

    return yaw_control



def navigate_pathlen_1(targets, current_target_ix, next_target_ix, ellipse_graph, M_list, center_list, viewpoint_ref, shape_matrices, offsets, ellipse_path):
    # We are in the same ellipse:  viewpoint -> waypoint
    # we are in the same ellipse as the viewpoint
    # wp_now is the target's predicted trajectory
    # wp_next is the intersection waypoint of the path to the following assignment

    wp_now = predict_target(targets.agent_list[current_target_ix])
    #ellipse_path_next, wp_next, shape_matrices_next, offsets_next = find_ellipsoid_path(ellipse_graph, M_list, center_list, viewpoint_ref, compute_viewpoint_ref(targets.agent_list[next_target_ix]))
    ellipse_path_next, wp_next, shape_matrices_next, offsets_next, _ = find_ellipsoid_path_weighted(ellipse_graph_weighted, M_list, center_list, viewpoint_ref, compute_viewpoint_ref(targets.agent_list[next_target_ix]))
    if len(wp_next) > 0:
        wp_next = np.hstack((wp_next[0], [0]))
        B = shape_matrices_next[1]
        b = offsets_next[1]
    else:
        wp_next = targets.agent_list[next_target_ix].unicycle_state[:3]
        B = shape_matrices[0]
        b = offsets[0]
    A = shape_matrices[0]
    a = offsets[0]

    ell1_ix = ellipse_path[0]
    if len(ellipse_path_next) > 0:
        ell2_ix = ellipse_path_next[0]
    else:
        ell2_ix = -1

    return wp_now, wp_next, A, B, a, b, ell1_ix, ell2_ix

def navigate_pathlen_2(waypoints, current_target_ix, targets, shape_matrices, offsets, ellipse_path):
    # 2) we are in the ellipse before the final one
    # wp_now is the intersection waypoint
    # wp_next is the viewpoint reference at the target
    wp_now = np.tile(np.hstack((waypoints[0], [0])), (40,1))
    wp_next = targets.agent_list[current_target_ix].unicycle_state[:3]
    A = shape_matrices[0]
    a = offsets[0]
    B = shape_matrices[1]
    b = offsets[1]
    ell1_ix = ellipse_path[0]
    ell2_ix = ellipse_path[1]

    return wp_now, wp_next, A, B, a, b, ell1_ix, ell2_ix

def navigate_pathlen_3plus(waypoints, shape_matrices, offsets, ellipse_path):
    # 3) we are further from the target viewpoint
    # wp_now is the first intersection waypoint
    # wp_next is the following intersection waypoint

    wp_now = np.tile(np.hstack((waypoints[0], [0])), (40,1))
    wp_next = np.hstack((waypoints[1], [0]))
    A = shape_matrices[0]
    a = offsets[0]
    B = shape_matrices[1]
    b = offsets[1]
    ell1_ix = ellipse_path[0]
    ell2_ix = ellipse_path[1]
    return wp_now, wp_next, A, B, a, b, ell1_ix, ell2_ix


def step_tracker(tracker, assignment_type, targets, targets_responsible, time_since_visited, mpc_guess, pre_assignments):


    if len(targets_responsible) > 0:
        # 1. from the assigned targets, plan which one to visit
        ix_map = np.arange(len(targets.agent_list))[targets_responsible]
        #target_positions = targets.agent_list[targets_responsible]
        target_positions = targets.state[targets_responsible,:2]
        visit_staleness = time_since_visited[targets_responsible]
        #distances = np.linalg.norm(target_positions - tracker.state[:2], axis=1) # l2 distance
        distances = [find_ellipsoid_path_weighted(ellipse_graph_weighted, M_list, center_list, tracker.state[:2], target_positions[ix])[-1] for ix in range(len(target_positions))] # obstacle-aware distance

        staleness_weight = 1
        visit_costs = distances - staleness_weight * visit_staleness
        assignments = ix_map[np.argsort(visit_costs)]

        # 2. Find path from current location to target

        # get the current and next target assignment
        current_target_ix = assignments[0]
        if len(assignments) > 1:
            next_target_ix = assignments[1]
        else:
            next_target_ix = assignments[0]

        # 1.) We are in the same ellipse:  viewpoint -> waypoint
        # 2.) We are in the ellipse before the target: waypoint -> viewpoint
        # 3.) We are further from target: waypoint -> waypoint
        # For 2 and 3, the weight switchover is determined by l2 distance from waypoint
        # For 1, weight switchover is determined by check_view
        viewpoint_ref = compute_viewpoint_ref(targets.agent_list[current_target_ix])
        #ellipse_path, waypoints, shape_matrices, offsets = find_ellipsoid_path(ellipse_graph, M_list, center_list, tracker.unicycle_state[:2], viewpoint_ref)
        ellipse_path, waypoints, shape_matrices, offsets, _ = find_ellipsoid_path_weighted(ellipse_graph_weighted, M_list, center_list, tracker.unicycle_state[:2], viewpoint_ref)
        pathlen = len(ellipse_path)

        if pathlen == 0:
            raise Exception('Pathlen 0, should not happen!')
        elif pathlen == 1:
            wp_now, wp_next, A, B, a, b, ell1_ix, ell2_ix = navigate_pathlen_1(targets, current_target_ix, next_target_ix, ellipse_graph, M_list, center_list, viewpoint_ref, shape_matrices, offsets, ellipse_path)
            travel_weight = 0

        elif pathlen == 2:
            wp_now, wp_next, A, B, a, b, ell1_ix, ell2_ix = navigate_pathlen_2(waypoints, current_target_ix, targets, shape_matrices, offsets, ellipse_path)
            travel_weight = 1
        else:
            # 3) we are further from the target viewpoint
            # wp_now is the first intersection waypoint
            # wp_next is the following intersection waypoint
            wp_now, wp_next, A, B, a, b, ell1_ix, ell2_ix = navigate_pathlen_3plus(waypoints, shape_matrices, offsets, ellipse_path)
            travel_weight = 1
            
        if not sorted([ell1_ix, ell2_ix]) == tracker.params.ellipse_switch_pair:
            tracker.params.ellipse_switch_pair = sorted([ell1_ix, ell2_ix])
            tracker.params.ellipse_switch_ix = 40
            tracker.params.switch_ix = 39

        ellipse_switch_ix = tracker.params.ellipse_switch_ix

        #print('Viz ref: ', viewpoint_ref)
        #print('ellipse switch index: ', ellipse_switch_ix)

        ubg[:ellipse_switch_ix, 5] = 1
        ubg[:ellipse_switch_ix, 6] = np.inf
        ubg[ellipse_switch_ix:, 5] = np.inf
        ubg[ellipse_switch_ix:, 6] = 1

        #weights_1[0:ellipse_switch_ix] = 1
        #weights_1[ellipse_switch_ix:] = 0
        #weights_2[0:ellipse_switch_ix] = 0
        #weights_2[ellipse_switch_ix:] = 1

        switch_ix = tracker.params.switch_ix
        weights_1[0:switch_ix] = 1
        weights_1[switch_ix:] = 0
        weights_2[0:switch_ix] = 0
        weights_2[switch_ix:] = 1

        # Solve MPC
        target_prediction = predict_target(targets.agent_list[current_target_ix])
        sol = solver_comp(x0=mpc_guess, lbx=lbw, ubx=ubw, lbg=lbg.flatten(), ubg=ubg.flatten(), p=cd.vertcat(tracker.unicycle_state_mpc, wp_now.flatten(), wp_next, weights_1, weights_2, A.flatten(), B.flatten(), a, b, travel_weight))
        w0 = sol['x']

        controls = np.array(sol['x'][:2]).flatten()
        tracker.control[:2] = controls[:]
        trackers.synchronize_state()

        if travel_weight == 0:
            tracker.params.switch_ix = update_switch_viewpoint(wp_now, w0, switch_ix)
        else:
            tracker.params.switch_ix = update_switch_waypoint(wp_now, w0, switch_ix)

        traj_pad = np.hstack([np.zeros(5), np.array(w0).flatten()])
        traj_mat = np.reshape(traj_pad, (41, 7))

        for jx in range(1, len(traj_mat)):
            pos = traj_mat[jx, :2]
            rad = (pos - b) @ B @ (pos - b)[:,None]
            if rad < 1:
                ellipse_switch_ix = min(len(traj_mat), jx + 1)
                break
        tracker.params.ellipse_switch_ix = ellipse_switch_ix
    else:
        w0 = .1*cd.vertcat(np.random.random(282))
        ell1_ix = -1
        ell2_ix = -1
        assignments = []


    return w0, assignments, [ell1_ix, ell2_ix]



def divide_targets_nodecomposition_obsaware(targets, trackers):
    positions_to_visit = targets.pose[:, :2]

    target_to_tracker = np.array([np.argmin([find_ellipsoid_path_weighted(ellipse_graph_weighted, M_list, center_list, trackers.agent_list[jx].state[:2], targets.agent_list[ix].state[:2]) for jx in range(len(trackers.agent_list))]) for ix in range(n_targets)])

    targets_per_tracker = [[jx for jx in np.where(target_to_tracker == ix)[0]] for ix in range(n_trackers)] # list of targets for each 
    n_per_tracker = [len(t) for t in targets_per_tracker]
    # if some trackers have no local targets, pursue a target that is closer to a different tracker
    if max(n_per_tracker) > 0:
        for ix in range(len(targets_per_tracker)):
            if len(targets_per_tracker[ix]) == 0:
                target_dists = [np.linalg.norm(targets.agent_list[jx].state[:2] - trackers.agent_list[ix].state[:2]) for jx in range(n_targets)]
                targets_per_tracker[ix] = [np.argmin(target_dists)]
    return targets_per_tracker


def divide_targets_nodecomposition(targets, trackers):
    positions_to_visit = targets.pose[:, :2]
    target_to_tracker = np.array([np.argmin([np.linalg.norm(targets.agent_list[ix].state[:2] - trackers.agent_list[jx].state[:2]) for jx in range(len(trackers.agent_list))]) for ix in range(n_targets)])
    targets_per_tracker = [[jx for jx in np.where(target_to_tracker == ix)[0]] for ix in range(n_trackers)] # list of targets for each 
    n_per_tracker = [len(t) for t in targets_per_tracker]
    # if some trackers have no local targets, pursue a target that is closer to a different tracker
    if max(n_per_tracker) > 0:
        for ix in range(len(targets_per_tracker)):
            if len(targets_per_tracker[ix]) == 0:
                target_dists = [np.linalg.norm(targets.agent_list[jx].state[:2] - trackers.agent_list[ix].state[:2]) for jx in range(n_targets)]
                targets_per_tracker[ix] = [np.argmin(target_dists)]
    return targets_per_tracker

def divide_targets_ellipsoid(targets, responsible_ellipsoids):
    indices = list(range(len(responsible_ellipsoids)))
    random.shuffle(indices)
    target_ix_list = list(range(len(targets.agent_list)))
    targets_per_tracker = [[] for _ in range(len(responsible_ellipsoids))]
    for tracker_ix in indices:
        target_ix_list_copy = target_ix_list.copy()
        for target_ix in target_ix_list_copy.copy():
            target = targets.agent_list[target_ix]
            tpos = target.unicycle_state[:2]
            for ell in responsible_ellipsoids[tracker_ix]:
                M, c = ell
                dist =  (tpos - c) @ M @ (tpos - c)[:,None]
                if dist < 1.:
                    targets_per_tracker[tracker_ix].append(target_ix)
                    target_ix_list_copy.remove(target_ix)
                    break
    return targets_per_tracker


weights_1 = cd.DM.ones(40)
weights_2 = cd.DM.zeros(40)

time_since_visited_list = np.zeros(n_targets)
log_filename = 'logs/%s_%s_%d_trackers_%d_targets_%f.txt' % (env_type, HLP_TYPE, n_trackers, n_targets, time.time())
time_visited_log = open(log_filename, 'w')

def update(i):
    global current_target_indices, w0, switch_ix, assignment_ix, time_since_visited_list
    print('Current Time Index: ', i)
    print(time_since_visited_list)

    line_to_write = (('%d,'*n_targets)[:-1] + '\n') % tuple(time_since_visited_list.tolist())
    time_visited_log.write(line_to_write)

    time_since_visited_list += 1

    if HLP_TYPE == 'no_decomposition':
        # "dynamic voronoi"
        targets_per_tracker = divide_targets_nodecomposition(targets, trackers)
    elif HLP_TYPE == 'no_decomposition_obsaware':
        targets_per_tracker = divide_targets_nodecomposition_obsaware(targets, trackers)
    elif HLP_TYPE == 'static_voronoi':
        # do an initial, static, obstacle-unaware voronoi decomposition
        raise NotImplementedError('static voronoi not implemented')
    elif HLP_TYPE == 'ellipsoids':
        targets_per_tracker = divide_targets_ellipsoid(targets, ellipsoids_per_tracker)
    else:
        raise Exception('Unknown High Level Planning type %s' % (HLP_TYPE))

    t0 = time.time()
    ellipse_indices = []
    for (ix, tracker) in enumerate(trackers.agent_list):
        # still need to divide targets
        trajectory, assignments, ellipses = step_tracker(tracker, assignment_type, targets, targets_per_tracker[ix], time_since_visited_list, mpc_guesses[ix], last_assignment[ix])
        ellipse_indices += ellipses
        mpc_guesses[ix] = trajectory
        assignment_ix[ix] = assignments
        last_assignment[ix] = assignments

        if len(assignments) > 0:
            current_target_indices[ix] = assignments[0]
        else:
            current_target_indices[ix] = None
        #trajectories[ix] = trajectory
        trajectory = np.array(trajectory)
        if len(trajectory) > 0:
            trajectories[ix] = np.reshape(np.hstack([np.zeros(5), np.array(trajectory).flatten()]), (41,7))

    update_agents(trackers, 5.0/40)

    #print('updating target goals')
    for t in targets.agent_list:
        #print('waypoints: ', t.waypoints)
        waypoint_tolerance = 1
        if len(t.waypoints) > 0:
            if np.linalg.norm(t.unicycle_state[:2] - t.waypoints[0]) < waypoint_tolerance:
                if len(t.waypoints) > 1:
                    t.waypoints = t.waypoints[1:]
                else:
                    t.waypoints = resample_target_goal(t.unicycle_state[:2], env)
                    #t.waypoints = [resample_target_goal(t.unicycle_state[:2], env)[-1]]
        else:
            t.waypoints = resample_target_goal(t.unicycle_state[:2], env)
            #t.waypoints = [resample_target_goal(t.unicycle_state[:2], env)[-1]]
            #print(t.waypoints)
        t.control[0] = 0.3 # constant velocity
        t.control[1] = 6*pd_control_update(t, t.waypoints[0])
    update_targets(targets, 5.0/40)
    targets.synchronize_state()

    for (ix, ta) in enumerate(targets.agent_list):
        for tr in trackers.agent_list:
            if check_visited_view(tr, ta):
                time_since_visited_list[ix] = 0
                continue
            

    # Update plotting
    if not args.plot_mode == 'none':
        update_plot_target_group(targets, scats)
        update_plot_tracker_group(trackers, trajectories, current_target_indices, targets, assignment_ix, scats_tracker)
        #print('Ellipse indices to plot: ', ellipse_indices)
        update_plot_environment(env_cxt, ellipse_indices)

        artists = cxt_to_artists(scats, scats_tracker) # need to return the artists so the animation can update
        return artists

if args.plot_mode == 'none':
    for ix in range(args.n_steps):
        update(ix)
elif args.plot_mode == 'show':
    ani = animation.FuncAnimation(fig, update, frames = np.arange(5000), interval=400, blit=False, repeat=False)
    plt.show()
elif args.plot_mode == 'save':
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.FuncAnimation(fig, update, frames=np.arange(args.n_steps), blit=True, save_count=args.n_steps)
    ani.save('videos/%s' % args.animation_name, writer=writer)



