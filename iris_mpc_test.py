#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as cd
from shapely import geometry
import irispy

from environment import Environment
from agents import AgentGroup, DefaultTrackerParams, DefaultTargetParams
from visualization import initial_plot_tracker_group, initial_plot_target_group, plot_environment, update_plot_tracker_group, cxt_to_artists
#from utils import ellipsoids_intersect, get_path
from dynamics import update_agents

from mpc_obs_functions import *

np.random.seed(1)

obs1 = np.array([ [0, 1], [3, 1], [3,4], [0,4] ])
obs2 = np.array([ [3,3], [5,3], [5,4], [3,4] ])
obs3 = np.array([ [5,1], [8,1], [8,4], [5, 4] ])

obs4 = np.array([ [0, -1], [8, -1], [8, -2], [0, -2] ])
obs5 = np.array([ [3.75, -1], [4.25, -1], [4.25, 1], [3.75, 1] ])
obstacles = [obs1, obs2, obs3, obs4, obs5]

env = Environment(obstacles, [[-10, -10], [10, 10]])

#ref_path = np.array([ [0,0], [3.5, 0], [3.5, 1], [3.5, 2], [4.5, 2], [4.5, 1], [4.5, 0], [8, 0] ])

region_list, c_inv_sq_list, c_list, d_list = construct_ellipse_space(env)

ellipse_connectivity = construct_ellipse_topology(c_inv_sq_list, d_list)

start = np.array([-4,-5])
end = np.array([8, 0])


ellipse_path, waypoints, shape_matrices, offsets = find_ellipsoid_path(ellipse_connectivity, c_inv_sq_list, d_list, start, end)

fig, ax = plt.subplots()
for r in [region_list[i] for i in ellipse_path]:
    r.ellipsoid.draw()

n_trackers = 1
trackers = AgentGroup(n_trackers, [-5,-5,-5,], [5,5,5], DefaultTrackerParams())
trackers.unicycle_state[0,:2] = np.array([-4,-5])
trackers.unicycle_state[0,5] = 0.1
trackers.synchronize_state()

targets = AgentGroup(1, [-5,-5,-5,], [5,5,5], DefaultTargetParams())
targets.unicycle_state[0, :2] = np.array([8, 0])
targets.synchronize_state()


scats= initial_plot_target_group(ax, targets)
scats_tracker = initial_plot_tracker_group(ax, trackers)
plot_environment(ax, env)

plt.scatter(waypoints[:,0], waypoints[:,1])

#plt.plot(xpoly_plot, ypoly_plot)

#for seed_point in ref_path:
#    #seed_point = np.array([4.0, 2.0])
#    region = call_irispy(env, seed_point)
#    print(region.ellipsoid.getC()) # shape matrix
#    print(region.ellipsoid.getD()) # center
#    region.polyhedron.draw(edgecolor='g')
#    region.ellipsoid.draw()
#plt.show()


solver_comp = cd.nlpsol('solver_iris', 'ipopt', './nlp_iris_2.so', {'print_time':0, 'ipopt.print_level' : 0, 'ipopt.max_cpu_time': .6})

w0 = cd.vertcat(np.random.random(282))
bounds = cd.inf
lbw = np.array([-bounds, -bounds, -cd.inf, 0, -cd.pi/2.0, -3, -3])
ubw = np.array([bounds, bounds, cd.inf, 3, cd.pi/2.0, 3, 3])

lbw = np.tile(lbw, (41, 1)).flatten()[5:]
ubw = np.tile(ubw, (41, 1)).flatten()[5:]

lbg_base = np.array([0.,0,0,0,0,0,0])
ubg_base = np.array([0.,0,0,0,0,0,0])

lbg = np.tile(lbg_base, (40, 1))
ubg = np.tile(ubg_base, (40, 1))

mpc_guesses = [w0] * n_trackers 
weights_1 = cd.DM.ones(40)
weights_2 = cd.DM.ones(40)

weights_1[38:] = 0
weights_2[:38] = 0


waypoint_ix = 0
switch_ix = 40

def update(ix):
    global waypoint_ix, switch_ix
    print(ix)

    #pos = trackers.agent_list[0].unicycle_state[:2]
    #theta = trackers.agent_list[0].unicycle_state[2]
    #rev = np.array([np.cos(theta), np.sin(theta)])
    #perp = np.array([-np.sin(theta), np.cos(theta)])
    #new_obs = np.array([pos - 1.5*rev + 5*perp, pos - 1.5*1.5*rev + 5*perp, pos - 1.5*1.5*rev - 5*perp, pos - 1.5*rev - 5*perp])
    #env_temp = Environment(obstacles + [new_obs], [[-10, -10], [10, 10]])
    #print(len(env_temp.obstacles))
    ##plot_environment(ax, env_temp)
    ##region = call_irispy(env_temp, trackers.agent_list[0].unicycle_state[:2])
    ##region.ellipsoid.draw()
    #ell_agent_d = region.ellipsoid.getD()
    #c_ell = region.ellipsoid.getC()
    #c_inv = np.linalg.inv(c_ell)
    #ell_agent_c = c_inv @ c_inv


    # Solve MPC
    #if np.linalg.norm(centers[waypoint_ix] - trackers.agent_list[0].unicycle_state[:2]) < .3:
    if np.linalg.norm(waypoints[waypoint_ix] - trackers.agent_list[0].unicycle_state[:2]) < .3:
        if waypoint_ix < len(waypoints)-1:
            waypoint_ix += 1
        switch_ix = 40
    if waypoint_ix == len(waypoints) - 1:
        travel_weight = 0
    else:
        travel_weight = 1
    next_ix = min(waypoint_ix + 1, len(waypoints)-1)

    if travel_weight == 1:
        target_prediction = np.tile(np.hstack((waypoints[waypoint_ix], [0])), (40, 1))
        next_target = np.hstack([waypoints[next_ix], [0]])
    else:
        target_prediction = np.tile(targets.agent_list[0].unicycle_state[:3], (40, 1))
        next_target = targets.agent_list[0].unicycle_state[:3]


    A = shape_matrices[waypoint_ix]
    B = shape_matrices[min(len(shape_matrices)-1, next_ix)]

    a = offsets[waypoint_ix]
    b = offsets[min(len(shape_matrices)-1, next_ix)]

    ubg[:switch_ix, 5] = 1
    ubg[:switch_ix, 6] = np.inf 
    ubg[switch_ix:, 5] = np.inf 
    ubg[switch_ix:, 6] = 1

    print('waypoint ix: ', waypoint_ix)
    print('waypoint_position:', waypoints[waypoint_ix])
    
    sol = solver_comp(x0=mpc_guesses[0], lbx=lbw, ubx=ubw, lbg=lbg.flatten(), ubg=ubg.flatten(), p=cd.vertcat(trackers.agent_list[0].unicycle_state_mpc, target_prediction.flatten(), next_target, weights_1, weights_2, A.reshape((4,1)), B.reshape((4,1)), a, b, travel_weight))
    w0 = sol['x']

    controls = np.array(sol['x'][:2]).flatten()
    trackers.agent_list[0].control[:2] = controls[:]
    trackers.synchronize_state()
    update_agents(trackers, 5.0/40)



    traj_2d = np.reshape(np.hstack((np.zeros(5), np.array(w0).flatten())), (41,7))
    ## update when we switch constraint ellipse
    for jx in range(1, traj_2d.shape[0]):
        pos = traj_2d[jx, :2]
        rad = (pos - b) @ B @ (pos - b)[:,None]
        if rad < 1:
            switch_ix = jx + 1
            break

    update_plot_tracker_group(trackers, [traj_2d], [0], targets, [[0]], scats_tracker, None)

    artists = cxt_to_artists(scats, scats_tracker)
    return artists

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, update, range(1, 2000), interval=200, blit=False)
#ani = animation.FuncAnimation(fig, update, frames=range(1,2000), blit=True, save_count=3000)
plt.show()
#ani.save('videos/temp.mp4', writer=writer)

