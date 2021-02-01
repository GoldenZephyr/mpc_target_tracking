# visualization.py -- Visualze targets and trackers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def cxt_to_artists(targets, trackers):
    artists = []
    artists += targets.target_scatter_list
    artists += targets.target_fov_list
    artists += trackers.tracker_scatter_list
    artists += trackers.track_list
    artists += trackers.traj_plots
    return artists

class TargetPlotCxt:
    def __init__(self):
        self.target_scatter_list = []
        self.target_fov_list = []

class TrackerPlotCxt:
    def __init__(self):
        self.tracker_scatter_list = []
        self.track_list = []
        self.traj_plots = []
        self.tsp_plots = []

def generate_triangle_pts(t):
    r = 2
    ang1 = t.orientation[2] + t.params.information_angle + t.params.information_tolerance
    ang2 = t.orientation[2] + t.params.information_angle - t.params.information_tolerance / 2.0

    pt0 = t.position[:2]
    pt1 = pt0 + np.array([r*np.cos(ang1), r*np.sin(ang1)])
    pt2 = pt0 + np.array([r*np.cos(ang2), r*np.sin(ang2)])

    return np.array([pt0, pt1, pt2])


def initial_plot_tracker_group(ax, trackers):
    scatter_list = []
    cxt = TrackerPlotCxt()
    for t in trackers.agent_list:
    
        l = ax.scatter(t.position[0], t.position[1], color='k')
        cxt.tracker_scatter_list.append(l)

        xs = [t.position[0], t.position[0] + 1]
        ys = [t.position[1], t.position[1] + 1]
        l2 = ax.plot(xs, ys, color='b')[0]
        cxt.track_list.append(l2)

        xs = np.zeros(41)
        ys = np.zeros(41)
        l3 = ax.plot(xs, ys, color='g')[0]
        cxt.traj_plots.append(l3)

        l4 = ax.plot(xs, ys, color='c', alpha=0.2)[0]
        cxt.tsp_plots.append(l4)

    return cxt

def initial_plot_target_group(ax, group):
    scatter_list = []
    cxt = TargetPlotCxt()
    for t in group.agent_list:
        l = ax.scatter(t.position[0], t.position[1])
        cxt.target_scatter_list.append(l)

        tri_corners = generate_triangle_pts(t)
        patch = patches.Polygon(tri_corners, closed=True, fc='r', ec='r', alpha=0.2)
        ax.add_patch(patch)
        cxt.target_fov_list.append(patch)
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])

    return cxt

def update_plot_tracker_group(group, trajectories, index_asgn, targets, tsp, cxt):
    for (ix, t) in enumerate(group.agent_list):
        cxt.tracker_scatter_list[ix].set_offsets(t.position[:-1])

        target_position = targets.agent_list[index_asgn[ix]].position[:2]
        tracker_position = t.position[:2]
        xs = [tracker_position[0], target_position[0]]
        ys = [tracker_position[1], target_position[1]]
        cxt.track_list[ix].set_data(xs, ys)

        vars = np.hstack([np.zeros(5), np.array(trajectories[ix]).flatten()])
        vars_2d = np.reshape(vars, (41,7))
        xs = vars_2d[1:,0]
        ys = vars_2d[1:,1]
        cxt.traj_plots[ix].set_data(xs, ys)

        xs = tsp[:, 0]
        ys = tsp[:, 1]
        cxt.tsp_plots[ix].set_data(xs, ys)

def update_plot_target_group(group, cxt):
    for (ix, t) in enumerate(group.agent_list):
        cxt.target_scatter_list[ix].set_offsets(t.position[:-1])

        tri_corners = generate_triangle_pts(t)
        cxt.target_fov_list[ix].set_xy(tri_corners)
