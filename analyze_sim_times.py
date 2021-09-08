#!/usr/bin/python3

import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_times(fn):
    data = np.genfromtxt(fn, delimiter=',')
    meantime = np.mean(data)
    meansqtime = np.mean(data**2)
    maxtime = np.max(data)
    return meantime, meansqtime, maxtime

def print_stats(stats, alg_name, env_name):
    meantime_mean = np.mean([s[0] for s in stats])
    meantime_std = np.std([s[0] for s in stats])
    meansqtime_mean = np.mean([s[1] for s in stats])
    meansqtime_std = np.std([s[1] for s in stats])
    maxtime_mean = np.mean([s[2] for s in stats])
    maxtime_std = np.std([s[2] for s in stats])

    print('%s env, %s meantime: %f +/- %f (1std)' % (env_name, alg_name, meantime_mean, meantime_std))
    print('%s env, %s meansqtime: %f +/- %f (1std)' % (env_name, alg_name, meansqtime_mean, meansqtime_std))
    print('%s env, %s maxtime: %f +/- %f (1std)' % (env_name, alg_name, maxtime_mean, maxtime_std))
    print('')

def plot_mean_timing_comparison(ell_stats, sv_stats, nd_obs_stats, nd_stats, env):
    x = [2, 4, 6, 8]
    fig,ax = plt.subplots()
    ax.scatter([2]*len(ell_stats[env]), [s[0] for s in ell_stats[env]])
    ax.scatter([4]*len(sv_stats[env]), [s[0] for s in sv_stats[env]])
    ax.scatter([6]*len(nd_obs_stats[env]), [s[0] for s in nd_obs_stats[env]])
    ax.scatter([8]*len(nd_stats[env]), [s[0] for s in nd_stats[env]])
    ax.set_xticks(x)
    xlabels =['Static Ellipsoid Decomposition (ours)', 'Static Voronoi', 'Dynamic Ellipsoid Decomposition', 'Dynamic Voronoi']
    ax.set_xticklabels(xlabels, fontsize=18)
    ax.set_ylabel('Mean time between target visits')
    plt.title('Mean time between visits, %s domain' % env)
    plt.show()


def plot_max_timing_comparison(ell_stats, nv_stats, nd_obs_stats, nd_stats, env):
    x = [2, 4, 6, 8]
    fig,ax = plt.subplots()
    ax.scatter([2]*len(ell_stats[env]), [s[2] for s in ell_stats[env]])
    ax.scatter([4]*len(nv_stats[env]), [s[2] for s in nv_stats[env]])
    ax.scatter([6]*len(nd_obs_stats[env]), [s[2] for s in nd_obs_stats[env]])
    ax.scatter([8]*len(nd_stats[env]), [s[2] for s in nd_stats[env]])
    ax.set_xticks(x)
    xlabels =['Static Ellipsoid Decomposition (ours)', 'Static Voronoi', 'Dynamic Ellipsoid Decomposition', 'Dynamic Voronoi']
    ax.set_xticklabels(xlabels, fontsize=18)
    ax.set_ylabel('Max time between target visits')
    plt.title('Max time between visits, %s domain' % env)
    plt.show()


logdir = 'full_logs'
files = os.listdir(logdir)

domain_names = ['blocks', 'forest_']

ellipsoids_files = {}
no_decomp_obs = {}
no_decomp = {}
static_voronoi = {}
for d in domain_names:
    ellipsoids_files[d] = [os.path.join(logdir, f) for f in files if 'ellipsoids' in f and d in f]
    no_decomp_obs[d] = [os.path.join(logdir, f) for f in files if 'no_decomposition_obsaware' in f and d in f]
    no_decomp[d] = [os.path.join(logdir, f) for f in files if 'no_decomposition' in f and 'obsaware' not in f and d in f]
    static_voronoi[d] = [os.path.join(logdir, f) for f in files if 'static_voronoi' in f and d in f]


ellipsoids_stats = {}
no_decomp_obs_stats = {}
no_decomp_stats = {}
static_voronoi_stats = {}
for d in domain_names:
    ellipsoids_stats[d] = [analyze_times(f) for f in ellipsoids_files[d]]
    no_decomp_obs_stats[d] = [analyze_times(f) for f in no_decomp_obs[d]]
    no_decomp_stats[d] = [analyze_times(f) for f in no_decomp[d]]
    static_voronoi_stats[d] = [analyze_times(f) for f in static_voronoi[d]]

print_stats(ellipsoids_stats['blocks'], 'Ellipsoids', 'blocks')
print_stats(no_decomp_obs_stats['blocks'], 'no_decomp_obs', 'blocks')
print_stats(no_decomp_stats['blocks'], 'no_decomp', 'blocks')



print_stats(ellipsoids_stats['forest_'], 'Ellipsoids', 'forest')
print_stats(static_voronoi_stats['forest_'], 'Static voronoi', 'forest')
print_stats(no_decomp_obs_stats['forest_'], 'no_decomp_obs', 'forest')
print_stats(no_decomp_stats['forest_'], 'no_decomp', 'forest')


plot_mean_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'blocks')
plot_mean_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'forest_')

plot_max_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'blocks')
plot_max_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'forest_')




