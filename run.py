#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import casadi as cd
from ilp import generate_assignments
from utils import check_view, update_switch, predict_target

#from targets import TargetGroup
from agents import AgentGroup, DefaultTargetParams, DefaultTrackerParams
from visualization import initial_plot_target_group, update_plot_target_group, initial_plot_tracker_group, update_plot_tracker_group, cxt_to_artists
from tsp_glue import solve_tsp

from dynamics import update_agents, step


solver_comp = cd.nlpsol('solver', 'ipopt', './nlp.so', {'print_time':0, 'ipopt.print_level' : 0, 'ipopt.max_cpu_time': 0.08})

n_targets = 20
#targets = TargetGroup(n_targets, [-5,-5,-5], [5,5,5])
targets = AgentGroup(n_targets, [-5,-5,-5], [5,5,5], DefaultTargetParams())
trackers = AgentGroup(2, [-5,-5,-5,], [5,5,5], DefaultTrackerParams())

#plt.ion()
fig, ax = plt.subplots()

scats = initial_plot_target_group(ax, targets)
scats_tracker = initial_plot_tracker_group(ax, trackers)

#plt.draw()
#plt.waitforbuttonpress()

current_target_ix = 0
switch_ix = 39

w0 = cd.vertcat(np.random.random(282))
bounds = cd.inf
lbw = np.array([-bounds, -bounds, -cd.inf, 0, -cd.pi/4.0, -1, -2])
ubw = np.array([bounds, bounds, cd.inf, 3, cd.pi/4.0, 1, 2])

lbw = np.tile(lbw, (41, 1)).flatten()[5:]
ubw = np.tile(ubw, (41, 1)).flatten()[5:]

lbg = np.zeros(40*5)
ubg = np.zeros(40*5)

mpc_guesses = [w0, w0]


assignment_ix = [0]*2
current_target_indices = [0]*2
trajectories = [0] * 2

#def check_view(trackers, targets, target_ix):
#    tracker = trackers.agent_list[0]
#    target = targets.agent_list[target_ix]
#
#    x_diff = tracker.unicycle_state[:2] - target.unicycle_state[:2]
#    x_dist = np.linalg.norm(x_diff)
#    x_diff_heading = x_diff / x_dist
#
#    target_heading = np.array([np.cos(target.unicycle_state[2]), np.sin(target.unicycle_state[2])])
#
#    cos_theta = np.dot(x_diff_heading, target_heading)
#    if cos_theta > (np.sqrt(3) / 2.0) and x_dist < 2:
#        return True
#    else:
#        return False

#def update_switch(trackers, targets, traj, current_target_ix, switch_ix):
#    traj_pad = np.hstack([np.zeros(5), np.array(traj).flatten()])
#    traj_mat = np.reshape(traj_pad, (41, 7))
#    pos = traj_mat[:, 0:2]
#    
#    target_state = targets.agent_list[current_target_ix].unicycle_state
#    target_pos = target_state[:2]
#    pos_diff = pos - target_pos
#    pos_diff_heading = pos_diff / np.linalg.norm(pos_diff)
#
#    target_heading = np.array([np.cos(target_state[2]), np.sin(target_state[2])])
#
#    cos_theta = np.dot(pos_diff_heading, target_heading)
#    new_ix = np.argmax(cos_theta > np.sqrt(3)/2.0)
#    if new_ix == 0:
#        return min(switch_ix + 3, 39)
#    else:
#        return new_ix + 1 # +1 is just a little buffer, not technically necessary
#
#def predict_target(target):
#    prediction = np.zeros((40, 3))
#    state = np.copy(target.unicycle_state) 
#    u = np.zeros(2)
#    p = target.params
#
#    prediction[0] = state[:3]
#    for ix in range(39):
#        state = step(state, u, p, 5.0/40)
#        prediction[ix+1] = state[:3]
#    return prediction

#def step_tracker(tracker, use_tsp, use_knn, targets, targets_responsible):
def step_tracker(tracker, use_tsp, use_knn, targets, targets_responsible, mpc_guess):
    #ix_map = np.arange(len(targets_responsible))[targets_responsible]
    if not use_tsp:
        if use_knn:
            positions_to_visit = targets.unicycle_state[targets_responsible, :2] 
            tracker_position = tracker.state[:2]
            deltas = positions_to_visit - tracker_position
            distances = np.linalg.norm(deltas, axis=1)
            k = 10
            k_closest_ix = np.argsort(distances)[:k]
            knn_map = np.arange(len(targets_responsible))[k_closest_ix]
            target_positions = targets.unicycle_state[targets_responsible[k_closest_ix]]
        else:
            target_positions = targets.unicycle_state[targets_responsible]
            knn_map = np.arange(len(target_positions))
        (path_assignments, path_positions) = generate_assignments(target_positions, targets.agent_list[0].params, tracker.state[:2])
        assignments = [targets_responsible[knn_map[p]] for p in path_assignments]
    else:
        remaining_target_positions = targets.pose[targets_responsible,:2]
        tsp_sol = solve_tsp(remaining_target_positions, tracker.state[:2], 'tracker_%d_tsp.tsp' % tracker.index)
        assignments = [targets_responsible[p] for p in tsp_sol]


    current_target_ix = assignments[0]
    if len(assignments) > 1:
        next_target_ix = assignments[1]
    else:
        next_target_ix = assignments[0]


    move_on = check_view(tracker, targets, current_target_ix) 

    if move_on:
        tracker.params.switch_ix = 39
        #targets_responsible[current_target_ix] = False

    switch_ix = tracker.params.switch_ix
    weights_1[0:switch_ix+1] = 1
    weights_1[switch_ix] = 0
    weights_2[0:switch_ix+1] = 0
    weights_2[switch_ix] = 1

    # Solve MPC
    target_prediction = predict_target(targets.agent_list[current_target_ix])
    sol = solver_comp(x0=mpc_guess, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(tracker.unicycle_state, target_prediction.flatten(), targets.agent_list[next_target_ix].unicycle_state[:3], weights_1, weights_2))
    w0 = sol['x']

    controls = np.array(sol['x'][7:9]).flatten()
    tracker.control[:] = controls[:]
    trackers.synchronize_state()
    #update_agents(trackers, 5.0/40)
   
    tracker.params.switch_ix = update_switch(targets, w0, current_target_ix, switch_ix) 

    return assignments, w0, move_on

    
weights_1 = cd.DM.ones(40)
#weights_1[:20] = 1
weights_2 = cd.DM.zeros(40)
#weights_2[20:] = 1

need_visit_list = [True] * n_targets

use_tsp = True
use_knn = True
def update(i):
    global current_target_indices, w0, switch_ix, assignment_ix

    #assigment_ix = [0]*2
    #current_target_ix = [0]*2
    #trajectories = [0] * 2

    ## need to do this for each tracker
    #if not use_tsp:
    #    if use_knn:
    #        positions_to_visit = targets.unicycle_state[need_visit_list, :2] 
    #        tracker_position = trackers.agent_list[0].state[:2]
    #        deltas = positions_to_visit - tracker_position
    #        distances = np.linalg.norm(deltas, axis=1)
    #        k = 10
    #        k_closest_ix = np.argsort(distances)[:k]
    #        knn_map = np.arange(len(ix_map))[k_closest_ix]
    #        target_positions = targets.unicycle_state[ix_map[k_closest_ix]]
    #    else:
    #        target_positions = targets.unicycle_state[need_visit_list]
    #        knn_map = np.arange(len(target_positions))
    #    (path_assignments, path_positions) = generate_assignments(target_positions, targets.agent_list[0].params, trackers.agent_list[0].state[:2])
    #    current_target_ix = ix_map[knn_map[path_assignments[0]]]
    #    if len(path_assignments) > 1:
    #        next_target_ix = ix_map[knn_map[path_assignments[1]]]
    #    else:
    #        next_target_ix = ix_map[knn_map[path_assignments[0]]]

    #remaining_target_positions = targets.pose[need_visit_list,:2]
    #if use_tsp:
    #    tsp_sol = solve_tsp(remaining_target_positions, trackers.agent_list[0].state[:2], 'tracker_1_tsp.tsp')
    #    current_target_ix = ix_map[tsp_sol[0]]

    #    if len(tsp_sol) > 1:
    #        next_target_ix = ix_map[tsp_sol[1]]
    #    else:
    #        next_target_ix = ix_map[tsp_sol[0]]


    #move_on = check_view(trackers.agent_list[0], targets, current_target_ix) 

    #if move_on:
    #    #current_target_ix += 1
    #    switch_ix = 39
    #    need_visit_list[current_target_ix] = False

    #weights_1[0:switch_ix+1] = 1
    #weights_1[switch_ix] = 0
    #weights_2[0:switch_ix+1] = 0
    #weights_2[switch_ix] = 1

    ## Solve MPC
    #target_prediction = predict_target(targets.agent_list[current_target_ix])
    #sol = solver_comp(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(trackers.agent_list[0].unicycle_state, target_prediction.flatten(), targets.agent_list[next_target_ix].unicycle_state[:3], weights_1, weights_2))
    #w0 = sol['x']

    #trajectories = [w0]
    #controls = np.array(sol['x'][7:9]).flatten()
    #trackers.agent_list[0].control[:] = controls[:]
    #trackers.synchronize_state()
   
    #switch_ix = update_switch(targets, w0, current_target_ix, switch_ix) 


    ix_map = np.arange(n_targets)[need_visit_list]
    positions_to_visit = targets.pose[need_visit_list, :2]
    target_to_tracker = np.array([np.argmin([np.linalg.norm(targets.agent_list[ix].state[:2] - trackers.agent_list[jx].state[:2]) for jx in range(len(trackers.agent_list))]) for ix in ix_map])
    print(target_to_tracker)

    n_trackers = len(trackers.agent_list)
    print(np.where(target_to_tracker == 0))
    targets_per_tracker = [[ix_map[jx] for jx in np.where(target_to_tracker == ix)[0]] for ix in range(n_trackers)] # list of targets for each 

    print(targets_per_tracker)
    for (ix, tracker) in enumerate(trackers.agent_list):
        # still need to divide targets
        assignments, trajectory, move_on = step_tracker(tracker, use_tsp, use_knn, targets, targets_per_tracker[ix], mpc_guesses[ix])
        mpc_guesses[ix] = trajectory
        assignment_ix[ix] = assignments
        current_target_indices[ix] = assignments[0]
        trajectories[ix] = trajectory
        if move_on:
            need_visit_list[assignments[0]] = False

    update_agents(trackers, 5.0/40)

    # update the targets with random motion
    update_agents(targets, 5.0/40)
    for t in targets.agent_list:
        t.linear_acceleration[0] =  (np.random.random() - 0.5)
        t.angular_acceleration[0] = 3 * (np.random.random() - 0.5)
    targets.synchronize_state()

    # Update plotting
    update_plot_target_group(targets, scats)
    update_plot_tracker_group(trackers, trajectories, current_target_indices, targets, assignment_ix, scats_tracker)
    #if use_tsp:
    #    update_plot_tracker_group(trackers, trajectories, [current_target_ix], targets, targets.pose[ix_map[tsp_sol], :2], scats_tracker)
    #else:
    #    update_plot_tracker_group(trackers, trajectories, [current_target_ix], targets, targets.pose[ix_map[knn_map[path_assignments]], :2], scats_tracker)
    artists = cxt_to_artists(scats, scats_tracker) # need to return the artists so the animation can update
    return artists

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, update, range(1, 2000), interval=100, blit=True)
#plt.show()
ani.save('videos/multi_tracker1.mp4', writer=writer)
#plt.show()
#plt.waitforbuttonpress()
