#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as cd
from shapely import geometry
import irispy
from scipy.sparse.csgraph import shortest_path

from environment import Environment
from agents import AgentGroup, DefaultTrackerParams, DefaultTargetParams
from visualization import initial_plot_tracker_group, initial_plot_target_group, plot_environment, update_plot_tracker_group, cxt_to_artists
from utils import ellipsoids_intersect, get_path, ellipsoids_intersect2
from dynamics import update_agents


def find_ellipse_intersection(A, B, a, b):
    lower = 0
    upper = 1
    while 1:
        lam = (lower + upper) / 2.
        e_lambda = lam * A + (1 - lam) * B
        m_lambda = np.linalg.inv(e_lambda) @ (lam * A @ a[:, None] + (1 - lam) * B @ b[:,None])
       
        print('a', a)
        print('m_lam', m_lambda)
        print(A)
        dist = (a[:,None] - m_lambda).T @ A @ (a[:,None] - m_lambda)
        in_A = dist < 1
        dist = (b[:,None] - m_lambda).T @ B @ (b[:,None] - m_lambda)
        in_B = dist < 1

        if in_A and not in_B:
            lower = lam
        elif in_B and not in_A:
            upper = lam
        elif in_A and in_B:
            break
        else:
            raise Exception('Ellipses do not overlap!')

    return np.squeeze(m_lambda)


def call_irispy(env, seed):
    obs = [arr.T for arr in env.obstacles]
    bounds = irispy.Polyhedron.fromBounds(*env.bounds)

    region = irispy.inflate_region(obs, seed, bounds)
    return region


np.random.seed(1)

obs1 = np.array([ [0, 1], [3, 1], [3,4], [0,4] ])
obs2 = np.array([ [3,3], [5,3], [5,4], [3,4] ])
obs3 = np.array([ [5,1], [8,1], [8,4], [5, 4] ])

obs4 = np.array([ [0, -1], [8, -1], [8, -2], [0, -2] ])
obs5 = np.array([ [3.75, -1], [4.25, -1], [4.25, 1], [3.75, 1] ])
obstacles = [obs1, obs2, obs3, obs4, obs5]
shapely_obstacles = [geometry.Polygon(o) for o in obstacles]
#obstacles = []
#for ix in range(30):
#    verts = np.random.rand(3,2) - 0.5
#    verts += 10*(np.random.rand(1,2) - 0.5)
#    obstacles.append(verts)
env = Environment(obstacles, [[-10, -10], [10, 10]])

ref_path = np.array([ [0,0], [3.5, 0], [3.5, 1], [3.5, 2], [4.5, 2], [4.5, 1], [4.5, 0], [8, 0] ])

x = np.linspace(-8, 8, 30)
y = np.linspace(-8, 8, 30)
xv, yv = np.meshgrid(x, y)
xv = xv.ravel()
yv = yv.ravel()
needed_points = [True] * len(xv)
vals = np.vstack([xv, yv]).T
shapely_points = [geometry.Point(p) for p in vals]
ix_list = np.arange(len(xv))
c_list = []
c_inv_sq_list = []
d_list = []
region_list = []
while(any(needed_points)):
    first_ix = np.argmax(needed_points)
    needed_points[first_ix] = False
    point_in_obstacle = [o.contains(shapely_points[first_ix]) for o in shapely_obstacles]
    if any(point_in_obstacle):
        continue
    seed_point = vals[first_ix]
    region = call_irispy(env, seed_point)
    d = region.ellipsoid.getD()
    c = region.ellipsoid.getC()
    c_inv = np.linalg.inv(c)
    c_inv_sq = c_inv @ c_inv
    c_inv_sq_list.append(c_inv_sq)
    region_list.append(region)
    c_list.append(c) # shape matrix
    d_list.append(d) # center
    ix_remaining = ix_list[needed_points]
    for ix in ix_remaining:
        dist = (vals[ix] - d) @ c_inv_sq @ (vals[ix] - d)[:,None]
        eps = .01 # need this because sometimes the ellipsoid fits the seed point *on* the ellipse boundary
        if dist < 1 + eps:
            needed_points[ix] = False
    #fig = plt.figure()
    #for r in region_list:
    #    r.ellipsoid.draw()
    #plot_environment(plt.gca(), env)
    #plt.show()

n = len(region_list) 
#ellipse_connectivity = np.eye(n)
ellipse_connectivity = np.zeros((n,n))
for ix in range(n):
    A = c_inv_sq_list[ix]
    a = d_list[ix]
    for jx in range(ix + 1, n):
        B = c_inv_sq_list[jx]
        b = d_list[jx]
        if ellipsoids_intersect2(A, B, a, b):
            ellipse_connectivity[ix, jx] = 1
            ellipse_connectivity[jx, ix] = 1
           
A0 = c_inv_sq_list[0]
print(A0)
a0 = d_list[0]
print(a0)
B14 = c_inv_sq_list[14]
print(B14)
b14 = d_list[14]
print(b14)
inter = ellipsoids_intersect2(A0, B14, a0, b14)
print('Ellipsoids intersect?', inter)

plt.figure()
plt.imshow(ellipse_connectivity)
plt.show() 

start = np.array([-4,-5])
end = np.array([8, 0])
#end = np.array([-3, 6])

for ix in range(len(region_list)):
    dist = (start - d_list[ix]) @ c_inv_sq_list[ix] @ (start - d_list[ix])[:,None]
    if dist < 1.001:
        start_ellipse_ix = ix
        break
for ix in range(len(region_list)):
    dist = (end - d_list[ix]) @ c_inv_sq_list[ix] @ (end - d_list[ix])[:,None]
    if dist < 1.001:
        end_ellipse_ix = ix
        break

print(start_ellipse_ix, end_ellipse_ix)

D, pred = shortest_path(ellipse_connectivity, directed=False, method='FW', return_predecessors=True)
print(D)
plt.figure()
plt.imshow(D)
plt.show()
ellipse_path = get_path(pred, start_ellipse_ix, end_ellipse_ix)
print(ellipse_path)
ellipse_pairs = [(ellipse_path[ix], ellipse_path[ix+1]) for ix in range(len(ellipse_path) - 1)]
waypoints = [find_ellipse_intersection(c_inv_sq_list[a], c_inv_sq_list[b], d_list[a], d_list[b]) for (a,b) in ellipse_pairs]
#centers = np.array([start] + [d_list[ix] for ix in ellipse_path] + [end])
#centers = np.array([start] + waypoints + [end])
centers = np.array(waypoints + [end])
shape_matrices = [c_inv_sq_list[ix] for ix in ellipse_path]
offsets = [d_list[ix] for ix in ellipse_path]

#A,B,C,D,E = shape_matrices
#a,b,c,d,e = offsets


dists = np.cumsum(np.hstack(([0], np.linalg.norm(np.diff(centers, axis=0), axis=1))))
dists = dists / dists[-1]

n_poly = 10
xnew = np.linspace(0,1,20)
#xpoly = np.polynomial.polynomial.Polynomial.fit(dists, centers[:,0], n_poly - 1)
#ypoly = np.polynomial.polynomial.Polynomial.fit(dists, centers[:,1], n_poly - 1)
xpoly = np.polynomial.polynomial.Polynomial.fit(xnew, np.interp(xnew, dists, centers[:,0]), n_poly - 1)
ypoly = np.polynomial.polynomial.Polynomial.fit(xnew, np.interp(xnew, dists, centers[:,1]), n_poly - 1)

xpoly_coef = xpoly.convert().coef[::-1]
ypoly_coef = ypoly.convert().coef[::-1]
print('fit info')
print(dists)
print(centers[:,0])
print(centers[:,1])
print('Coefficients')
print(xpoly_coef)
print(ypoly_coef)

xpoly_plot = xpoly(np.linspace(0,1,40))
ypoly_plot = ypoly(np.linspace(0,1,40))

#plt.figure()
#plt.scatter(dists,centers[:,0])
#plt.scatter(xnew, np.interp(xnew, dists, centers[:,0]))
#plt.klot(np.linspace(0,1,40), xpoly_plot)
#plt.show()



fig, ax = plt.subplots()
#ax.plot(ref_path[:,0], ref_path[:,1])
#for r in region_list:
#for r in [region_list[i] for i in [0, 14]]:
for r in [region_list[i] for i in ellipse_path]:
    r.ellipsoid.draw()

#plt.scatter(xv,yv)
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

print(centers)
plt.scatter(centers[:,0], centers[:,1])
#plt.plot(xpoly_plot, ypoly_plot)

#for seed_point in ref_path:
#    #seed_point = np.array([4.0, 2.0])
#    region = call_irispy(env, seed_point)
#    print(region.ellipsoid.getC()) # shape matrix
#    print(region.ellipsoid.getD()) # center
#    region.polyhedron.draw(edgecolor='g')
#    region.ellipsoid.draw()
#plt.show()


#solver_comp = cd.nlpsol('solver_iris', 'ipopt', './nlp_iris.so', {'print_time':0, 'ipopt.print_level' : 0, 'ipopt.max_cpu_time': 1})
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
#lbg = np.zeros(40*6)
#ubg = np.zeros(40*6)

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
    if np.linalg.norm(centers[waypoint_ix] - trackers.agent_list[0].unicycle_state[:2]) < .3:
        waypoint_ix += 1
        switch_ix = 40
    next_ix = min(waypoint_ix + 1, len(centers)-1)
    target_prediction = np.tile(np.hstack((centers[waypoint_ix], [0])), (40, 1))

    A = shape_matrices[waypoint_ix]
    B = shape_matrices[min(len(shape_matrices)-1, next_ix)]

    a = offsets[waypoint_ix]
    b = offsets[min(len(shape_matrices)-1, next_ix)]

    ubg[:switch_ix, 5] = 1
    ubg[:switch_ix, 6] = np.inf #np.inf
    ubg[switch_ix:, 5] = 9999 #np.inf
    ubg[switch_ix:, 6] = 1

    print('waypoint ix: ', waypoint_ix)
    print('waypoint_position:', centers[waypoint_ix])
    
    #sol = solver_comp(x0=mpc_guesses[0], lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(trackers.agent_list[0].unicycle_state, target_prediction.flatten(), target_prediction[0], weights_1, weights_2, xpoly_coef, ypoly_coef, A.reshape((4,1)), B.reshape((4,1)), C.reshape((4,1)), D.reshape((4,1)), E.reshape((4,1)), ell_agent_c.reshape((4,1)), a, b, c, d, e, ell_agent_d ))
    #sol = solver_comp(x0=mpc_guesses[0], lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(trackers.agent_list[0].unicycle_state_mpc, target_prediction.flatten(), np.hstack([centers[next_ix], [0]]), weights_1, weights_2, A.reshape((4,1)), B.reshape((4,1)), C.reshape((4,1)), D.reshape((4,1)), E.reshape((4,1)), a, b, c, d, e))
    sol = solver_comp(x0=mpc_guesses[0], lbx=lbw, ubx=ubw, lbg=lbg.flatten(), ubg=ubg.flatten(), p=cd.vertcat(trackers.agent_list[0].unicycle_state_mpc, target_prediction.flatten(), np.hstack([centers[next_ix], [0]]), weights_1, weights_2, A.reshape((4,1)), B.reshape((4,1)), a, b, 1))
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
#ani = animation.FuncAnimation(fig, update, range(1, 2000), interval=200, blit=False)
ani = animation.FuncAnimation(fig, update, frames=range(1,2000), blit=True, save_count=3000)
#plt.show()
ani.save('videos/temp.mp4', writer=writer)

