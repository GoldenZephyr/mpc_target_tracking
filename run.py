#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import casadi as cd

#from targets import TargetGroup
from agents import AgentGroup, DefaultTargetParams, DefaultTrackerParams
from visualization import initial_plot_target_group, update_plot_target_group, initial_plot_tracker_group, update_plot_tracker_group, cxt_to_artists

from dynamics import update_targets


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
lbw = np.array([-10, -10, -cd.inf, 0, -cd.pi/4.0, -1, -2])
ubw = np.array([10, 10, cd.inf, 3, cd.pi/4.0, 1, 2])

lbw = np.tile(lbw, (41, 1)).flatten()[5:]
ubw = np.tile(ubw, (41, 1)).flatten()[5:]

lbg = np.zeros(40*5)
ubg = np.zeros(40*5)

def check_view(trackers, targets, target_ix):
    tracker = trackers.agent_list[0]
    target = targets.agent_list[target_ix]

    x_diff = tracker.unicycle_state[:2] - target.unicycle_state[:2]
    x_diff_heading = x_diff / np.linalg.norm(x_diff)

    target_heading = np.array([np.cos(target.unicycle_state[2]), np.sin(target.unicycle_state[2])])

    cos_theta = np.dot(x_diff_heading, target_heading)
    if cos_theta > (np.sqrt(3) / 2.0):
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
        return min(switch_ix + 1, 39)
    else:
        return new_ix
    
weights_1 = cd.DM.ones(40)
#weights_1[:20] = 1
weights_2 = cd.DM.zeros(40)
#weights_2[20:] = 1

#for ix in range(1000):
def update(i):
    global current_target_ix, w0, switch_ix
    move_on = check_view(trackers, targets, current_target_ix) 

    if move_on:
        current_target_ix += 1
        switch_ix = 39

    weights_1[0:switch_ix+1] = 1
    weights_1[switch_ix] = 0
    weights_2[0:switch_ix+1] = 0
    weights_2[switch_ix] = 1

    sol = solver_comp(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(trackers.agent_list[0].unicycle_state, targets.agent_list[current_target_ix].unicycle_state[:3], targets.agent_list[current_target_ix + 1].unicycle_state[:3], weights_1, weights_2))
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
    update_plot_tracker_group(trackers, trajectories, [current_target_ix], targets, scats_tracker)
    #fig.canvas.draw_idle()
    #plt.pause(.05)

    artists = cxt_to_artists(scats, scats_tracker)
    return artists

ani = animation.FuncAnimation(fig, update, range(1, 200), interval=80, blit=False)
plt.show()
#plt.waitforbuttonpress()
