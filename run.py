#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import casadi as cd

#from targets import TargetGroup
from agents import AgentGroup, DefaultTargetParams, DefaultTrackerParams
from visualization import initial_plot_target_group, update_plot_target_group, initial_plot_tracker_group, update_plot_tracker_group, cxt_to_artists
from tsp_glue import solve_tsp

from dynamics import update_targets, step


solver_comp = cd.nlpsol('solver', 'ipopt', './nlp.so', {'print_time':0, 'ipopt.print_level' : 0, 'ipopt.max_cpu_time': 0.08})


#targets = TargetGroup(20, [-5,-5,-5], [5,5,5])
targets = AgentGroup(20, [-5,-5,-5], [5,5,5], DefaultTargetParams())
trackers = AgentGroup(1, [-5,-5,-5,], [5,5,5], DefaultTrackerParams())

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

def check_view(trackers, targets, target_ix):
    tracker = trackers.agent_list[0]
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

def update_switch(trackers, targets, traj, current_target_ix, switch_ix):
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
    
weights_1 = cd.DM.ones(40)
#weights_1[:20] = 1
weights_2 = cd.DM.zeros(40)
#weights_2[20:] = 1

need_visit_list = [True] * 20

#for ix in range(1000):
def update(i):
    global current_target_ix, w0, switch_ix


    remaining_target_positions = targets.pose[need_visit_list,:2]
    ix_map = np.arange(20)[need_visit_list]
    tsp_sol = solve_tsp(remaining_target_positions, trackers.agent_list[0].state[:2], 'tracker_1_tsp.tsp')
    current_target_ix = ix_map[tsp_sol[0]]
    if len(tsp_sol) > 1:
        next_target_ix = ix_map[tsp_sol[1]]
    else:
        next_target_ix = ix_map[tsp_sol[0]]

    move_on = check_view(trackers, targets, current_target_ix) 

    if move_on:
        #current_target_ix += 1
        switch_ix = 39
        need_visit_list[current_target_ix] = False

    weights_1[0:switch_ix+1] = 1
    weights_1[switch_ix] = 0
    weights_2[0:switch_ix+1] = 0
    weights_2[switch_ix] = 1

    target_prediction = predict_target(targets.agent_list[current_target_ix])
    #sol = solver_comp(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(trackers.agent_list[0].unicycle_state, targets.agent_list[current_target_ix].unicycle_state[:3], targets.agent_list[current_target_ix + 1].unicycle_state[:3], weights_1, weights_2))
    sol = solver_comp(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(trackers.agent_list[0].unicycle_state, target_prediction.flatten(), targets.agent_list[next_target_ix].unicycle_state[:3], weights_1, weights_2))
    w0 = sol['x']
    trajectories = [w0]
    controls = np.array(sol['x'][7:9]).flatten()
    trackers.agent_list[0].control[:] = controls[:]
    trackers.synchronize_state()
    update_targets(trackers, 5.0/40)
   
    switch_ix = update_switch(trackers, targets, w0, current_target_ix, switch_ix) 

    update_targets(targets, 5.0/40)
    for t in targets.agent_list:
        t.linear_acceleration[0] =  (np.random.random() - 0.5)
        t.angular_acceleration[0] = 3 * (np.random.random() - 0.5)
    targets.synchronize_state()
    update_plot_target_group(targets, scats)
    update_plot_tracker_group(trackers, trajectories, [current_target_ix], targets, targets.pose[ix_map[tsp_sol], :2], scats_tracker)
    #update_plot_tracker_group(trackers, trajectories, [current_target_ix], targets, targets.pose[current_target_ix + 1, :2], scats_tracker)
    #fig.canvas.draw_idle()
    #plt.pause(.05)

    artists = cxt_to_artists(scats, scats_tracker)
    return artists

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, update, range(1, 2000), interval=150, blit=True)
ani.save('im.mp4', writer=writer)
#plt.show()
#plt.waitforbuttonpress()
