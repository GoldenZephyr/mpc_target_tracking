# visualization.py -- Visualze targets and trackers

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


def cxt_to_artists(targets, trackers):
    artists = []
    artists += targets.target_scatter_list
    artists += targets.target_fov_list
    artists += targets.target_wp_list
    artists += trackers.tracker_scatter_list
    artists += trackers.track_list
    artists += trackers.traj_plots
    artists += trackers.heading_plots
    return artists


class TargetPlotCxt:
    def __init__(self):
        self.target_scatter_list = []
        self.target_fov_list = []
        self.target_wp_list = []


class TrackerPlotCxt:
    def __init__(self):
        self.tracker_scatter_list = []
        self.heading_plots = []
        self.track_list = []
        self.traj_plots = []
        self.tsp_plots = []

        self.mpcc_points = [] 

class EnvironmentPlotCxt:
    def __init__(self, ax, env, ellipsoid_shape, ellipsoid_center):
        self.ellipsoid_plots = []
        plot_environment(ax, env)
        t = np.linspace(0, 2*np.pi + .1, 100)
        x = np.array([np.cos(t), np.sin(t)])
        for ix in range(len(ellipsoid_shape)):
            C = ellipsoid_shape[ix]
            d = ellipsoid_center[ix]
            y = C @ x + d[:,None]
            xv = y[0,:]
            yv = y[1,:]
            l = plt.plot(xv, yv, color='b')[0]
            self.ellipsoid_plots.append(l)


def update_plot_environment(cxt, ellipses_to_show):
    for ix in range(len(cxt.ellipsoid_plots)):
        alpha = 0 if ix in ellipses_to_show else 0
        cxt.ellipsoid_plots[ix].set_alpha(alpha)
    

def plot_environment(ax, env):
    patch_list = []
    for vertex_list in env.obstacles:
        polygon = patches.Polygon(vertex_list, True)
        patch_list.append(polygon)

    #p = PatchCollection(patch_list, cmap=matplotlib.cm.jet, alpha=0.4)
    p = PatchCollection(patch_list, alpha=1)

    #colors = 100*np.random.rand(len(patch_list))
    colors = 100*np.ones(len(patch_list))
    p.set_array(np.array(colors))

    ax.add_collection(p)


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

        l5 = ax.scatter([], [], color='m')
        cxt.mpcc_points.append(l5)

        l6 = ax.plot([],[], color='k')[0]
        cxt.heading_plots.append(l6)

    return cxt

def initial_plot_target_group(ax, group):
    scatter_list = []
    cxt = TargetPlotCxt()
    for t in group.agent_list:
        l = ax.scatter(t.position[0], t.position[1], color='r')
        cxt.target_scatter_list.append(l)

        tri_corners = generate_triangle_pts(t)
        patch = patches.Polygon(tri_corners, closed=True, fc='r', ec='r', alpha=0.2)
        ax.add_patch(patch)
        cxt.target_fov_list.append(patch)

        #l = ax.scatter([0],[0], color='k')
        l = ax.scatter([],[], color='k')
        cxt.target_wp_list.append(l)
    ax.set_xlim([-15,15])
    ax.set_ylim([-15,15])

    return cxt

def update_plot_tracker_group(group, trajectories, index_asgn, targets, tsp_order, cxt, mpcc_points=None):
    for (ix, t) in enumerate(group.agent_list):
        cxt.tracker_scatter_list[ix].set_offsets(t.position[:-1])

        if index_asgn[ix] is None:
            target_position = t.position[:2]
        else:
            target_position = targets.agent_list[index_asgn[ix]].position[:2]
        tracker_position = t.position[:2]
        xs = [tracker_position[0], target_position[0]]
        ys = [tracker_position[1], target_position[1]]
        cxt.track_list[ix].set_data(xs, ys)

        if index_asgn[ix] is None or len(trajectories[ix]) == 0:
            xs = [t.position[0]]
            ys = [t.position[1]]
        else:
            #vars = np.hstack([np.zeros(5), np.array(trajectories[ix]).flatten()])
            #vars_2d = np.reshape(vars, (41,7))
            #xs = vars_2d[1:,0]
            #ys = vars_2d[1:,1]
            xs = trajectories[ix][1:, 0]
            ys = trajectories[ix][1:, 1]
        cxt.traj_plots[ix].set_data(xs, ys)

        tsp_pos = targets.pose[tsp_order[ix], :2]
        xs = tsp_pos[:, 0]
        ys = tsp_pos[:, 1]
        #cxt.tsp_plots[ix].set_data(xs, ys)


        if mpcc_points is not None:
            cxt.mpcc_points[ix].set_offsets(mpcc_points[ix])

        pos = t.position[:2]
        theta = t.unicycle_state[2]
        direction = np.array([np.cos(theta), np.sin(theta)])
        p1 = pos
        p2 = pos + direction
        pts = np.vstack([p1, p2])
        cxt.heading_plots[ix].set_data(pts[:,0], pts[:,1])


def update_plot_target_group(group, cxt):
    #print('\n\n===========')
    for (ix, t) in enumerate(group.agent_list):
        cxt.target_scatter_list[ix].set_offsets(t.position[:-1])

        tri_corners = generate_triangle_pts(t)
        cxt.target_fov_list[ix].set_xy(tri_corners)

        #cxt.target_wp_list[ix].set_offsets(t.waypoints[0])
        #print('t.waypoints[0]: ', t.waypoints[0])
        #cxt.target_wp_list[ix].set_offsets(np.array([3,3]))

    #print('\n\n===========')
