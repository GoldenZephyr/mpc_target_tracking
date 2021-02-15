#!/usr/bin/python3

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import casadi as cd
from ilp import generate_assignments
from utils import check_view, update_switch, predict_target, mvee, rollout

#from targets import TargetGroup
from agents import AgentGroup, DefaultTargetParams, DefaultTrackerParams
from visualization import initial_plot_target_group, update_plot_target_group, initial_plot_tracker_group, update_plot_tracker_group, cxt_to_artists, plot_environment
from tsp_glue import solve_tsp

from dynamics import update_agents, step

from mpc_obs_functions import *


no_video = False
n_targets = 10
n_trackers = 1
assignment_type = 'TSP'

keep_going = True
np.random.seed(4)


solver_comp = cd.nlpsol('solver', 'ipopt', './nlp_iris_2.so', {'print_time':0, 'ipopt.print_level' : 0, 'ipopt.max_cpu_time': 0.2})

#targets = TargetGroup(n_targets, [-5,-5,-5], [5,5,5])
targets = AgentGroup(n_targets, [-5,-5,-5], [5,5,5], DefaultTargetParams())
trackers = AgentGroup(n_trackers, [-5,-5,-5,], [5,5,5], DefaultTrackerParams())

#env = Environment([np.random.rand(5,2) for jx in range(5)])
env = construct_environment(10, 10)
region_list, M_list, center_list = construct_ellipse_space(env)
graph = construct_ellipse_topology(M_list, center_list)


fig, ax = plt.subplots()

scats = initial_plot_target_group(ax, targets)
scats_tracker = initial_plot_tracker_group(ax, trackers)

#plot_environment(ax, env)
#plt.show()

current_target_ix = 0
switch_ix = 39

w0 = cd.vertcat(np.random.random(282))
bounds = cd.inf
lbw = np.array([-bounds, -bounds, -cd.inf, 0, -cd.pi/4.0, -1, -2])
ubw = np.array([bounds, bounds, cd.inf, 3, cd.pi/4.0, 1, 2])

lbw = np.tile(lbw, (41, 1)).flatten()[5:]
ubw = np.tile(ubw, (41, 1)).flatten()[5:]

lbg_base = np.array([0.,0,0,0,0,0,0])
ubg_base = np.array([0.,0,0,0,0,0,0])

lbg = np.tile(lbg_base, (40, 1))
ubg = np.tile(ubg_base, (40, 1))


mpc_guesses = [w0] * n_trackers


assignment_ix = [0]*n_trackers
current_target_indices = [0]*n_trackers
trajectories = [0] * n_trackers
def step_tracker(tracker, assignment_type, targets, targets_responsible, mpc_guess):

    t0 = time.time()

    if 'ILP' in assignment_type:
        if 'KNN' in assignment_type:
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
    elif assignment_type == 'ELL':
        target_rollouts = np.zeros((len(targets_responsible), 2))
        for ix in range(len(targets_responsible)):
            _, p = rollout(targets.agent_list[targets_responsible[ix]].unicycle_state, targets.agent_list[targets_responsible[ix]].params, 100, .1)
            target_rollouts[ix] = p[-1]
        volumes = np.zeros(len(targets_responsible))
        for ix in range(len(targets_responsible)):
            d = np.linalg.norm(tracker.state[:2] - targets.unicycle_state[targets_responsible[ix], :2])
            t = d / tracker.params.max_velocity[0]
            #ix_other = targets_responsible[:ix] + targets_responsible[ix+1:]
            ix_other = [j for j in range(ix)] + [j for j in range(ix+1, len(targets_responsible))]
            positions_other = target_rollouts[ix_other, :2]
            A, c = mvee(positions_other)
            volumes[ix] = np.linalg.det(A)
            assignments = [targets_responsible[np.argmin(volumes)]]
    elif assignment_type == 'TSP':
        remaining_target_positions = targets.pose[targets_responsible,:2]
        tsp_sol = solve_tsp(remaining_target_positions, tracker.state[:2], 'tracker_%d_tsp.tsp' % tracker.index)
        assignments = [targets_responsible[p] for p in tsp_sol]
    else:
        raise Exception('Unknown planner type %s' % assignment_type)

    print('HLP took %f seconds' % (time.time() - t0))

    if len(assignments) > 0:
        current_target_ix = assignments[0]
        if len(assignments) > 1:
            next_target_ix = assignments[1]
        else:
            next_target_ix = assignments[0]


        move_on = check_view(tracker, targets, current_target_ix) 

        if move_on:
            tracker.params.switch_ix = 39

        switch_ix = tracker.params.switch_ix
        weights_1[0:switch_ix+1] = 1
        weights_1[switch_ix] = 0
        weights_2[0:switch_ix+1] = 0
        weights_2[switch_ix] = 1


        # 1.) We are in the same ellipse:  viewpoint -> waypoint
        # 2.) We are in the ellipse before the target: waypoint -> viewpoint
        # 3.) We are further from target: waypoint -> waypoint
        # For 2 and 3, the weight switchover is determined by l2 distance from waypoint
        # For 1, weight switchover is determined by check_view
        

        # Solve MPC
        target_prediction = predict_target(targets.agent_list[current_target_ix])
        sol = solver_comp(x0=mpc_guess, lbx=lbw, ubx=ubw, lbg=lbg.flatten(), ubg=ubg.flatten(), p=cd.vertcat(tracker.unicycle_state_mpc, target_prediction.flatten(), targets.agent_list[next_target_ix].unicycle_state[:3], weights_1, weights_2))
        w0 = sol['x']

        controls = np.array(sol['x'][7:9]).flatten()
        tracker.mpc_control[:] = controls[:]
        trackers.synchronize_state()
       
        tracker.params.switch_ix = update_switch(targets, w0, current_target_ix, switch_ix) 
    else:
        # there are no more targets assigned to this agent
        w0 = cd.vertcat(np.random.random(282))
        move_on = False

    return assignments, w0, move_on

    
weights_1 = cd.DM.ones(40)
#weights_1[:20] = 1
weights_2 = cd.DM.zeros(40)
#weights_2[20:] = 1

need_visit_list = [True] * n_targets

def gen():
    global keep_going
    i = 0
    while keep_going:
        i += 1
        print('\n\n\n',i,'\n\n\n')
        yield i

def update(i):
    global current_target_indices, w0, switch_ix, assignment_ix, keep_going

    ix_map = np.arange(n_targets)[need_visit_list]
    positions_to_visit = targets.pose[need_visit_list, :2]
    target_to_tracker = np.array([np.argmin([np.linalg.norm(targets.agent_list[ix].state[:2] - trackers.agent_list[jx].state[:2]) for jx in range(len(trackers.agent_list))]) for ix in ix_map])

    n_trackers = len(trackers.agent_list)
    targets_per_tracker = [[ix_map[jx] for jx in np.where(target_to_tracker == ix)[0]] for ix in range(n_trackers)] # list of targets for each 

    t0 = time.time()
    for (ix, tracker) in enumerate(trackers.agent_list):
        # still need to divide targets
        assignments, trajectory, move_on = step_tracker(tracker, assignment_type, targets, targets_per_tracker[ix], mpc_guesses[ix])
        mpc_guesses[ix] = trajectory
        assignment_ix[ix] = assignments
        if len(assignments) > 0:
            current_target_indices[ix] = assignments[0]
        else:
            current_target_indices[ix] = None
        #trajectories[ix] = trajectory
        trajectories[ix] = np.reshape(np.hstack([np.zeros(5), np.array(trajectory).flatten()]), (41,7))
        if move_on:
            need_visit_list[assignments[0]] = False
    #if all([a is None for a in current_target_indices]):
    if not any(need_visit_list):
        keep_going = False
        print('\n\n\nDone!!!\n\n\n')

    print('Total planning took %f seconds' % (time.time() - t0))

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

    artists = cxt_to_artists(scats, scats_tracker) # need to return the artists so the animation can update
    return artists

if no_video:
    for ix in range(2000):
        update(ix)
else:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #ani = animation.FuncAnimation(fig, update, range(1, 2000), interval=100, blit=True)
    ani = animation.FuncAnimation(fig, update, frames=gen, blit=True, save_count=3000)
    #plt.show()
    ani.save('videos/temp.mp4', writer=writer)
    #plt.show()
