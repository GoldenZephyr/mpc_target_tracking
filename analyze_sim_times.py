#!/usr/bin/python3

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

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
    ax.scatter([2]*len(ell_stats[env]), [s[0]/8. for s in ell_stats[env]])
    ax.scatter([4]*len(sv_stats[env]), [s[0]/8. for s in sv_stats[env]])
    ax.scatter([6]*len(nd_obs_stats[env]), [s[0]/8. for s in nd_obs_stats[env]])
    ax.scatter([8]*len(nd_stats[env]), [s[0]/8. for s in nd_stats[env]])
    ax.set_xticks(x)
    xlabels =['Static Ellipsoid Decomposition (ours)', 'Static Voronoi', 'Dynamic Ellipsoid Decomposition', 'Dynamic Voronoi']
    ax.set_xticklabels(xlabels, fontsize=18)
    ax.set_ylabel('Mean time between target visits')
    plt.title('Mean time between visits, %s domain' % env)
    plt.show()


def plot_max_timing_comparison(ell_stats, nv_stats, nd_obs_stats, nd_stats, env):
    x = [2, 4, 6, 8]
    fig,ax = plt.subplots()
    ax.scatter([2]*len(ell_stats[env]), [s[2]/8. for s in ell_stats[env]])
    ax.scatter([4]*len(nv_stats[env]), [s[2]/8. for s in nv_stats[env]])
    ax.scatter([6]*len(nd_obs_stats[env]), [s[2]/8. for s in nd_obs_stats[env]])
    ax.scatter([8]*len(nd_stats[env]), [s[2]/8. for s in nd_stats[env]])
    ax.set_xticks(x)
    xlabels =['Static Ellipsoid Decomposition (ours)', 'Static Voronoi', 'Dynamic Ellipsoid Decomposition', 'Dynamic Voronoi']
    ax.set_xticklabels(xlabels, fontsize=18)
    ax.set_ylabel('Max time between target visits')
    plt.title('Max time between visits, %s domain' % env)
    plt.show()


def plot_mean_timing_comparison_paper(ell_stats, sv_stats, nd_obs_stats, nd_stats, env):
    ell_data = [s[0]/8. for s in ell_stats[env]]
    sv_data = [s[0]/8. for s in sv_stats[env]]
    nd_obs_data = [s[0]/8. for s in nd_obs_stats[env]]
    nd_data = [s[0]/8. for s in nd_stats[env]]
    xlabels =['Static Ellipsoid Decomposition (ours)', 'Static Voronoi', 'Dynamic Ellipsoid Decomposition', 'Dynamic Voronoi']
    fig,ax = plt.subplots()
    plt.boxplot([ell_data, sv_data, nd_obs_data, nd_data], labels=xlabels)
    ax.set_ylabel('Mean time between target visits')
    plt.title('Mean time between visits, %s domain' % env)
    plt.show()

def plot_max_timing_comparison_paper(ell_stats, sv_stats, nd_obs_stats, nd_stats, env):
    ell_data = [s[2]/8. for s in ell_stats[env]]
    sv_data = [s[2]/8. for s in sv_stats[env]]
    nd_obs_data = [s[2]/8. for s in nd_obs_stats[env]]
    nd_data = [s[2]/8. for s in nd_stats[env]]
    xlabels =['Static Ellipsoid Decomposition (ours)', 'Static Voronoi', 'Dynamic Ellipsoid Decomposition', 'Dynamic Voronoi']
    fig,ax = plt.subplots()
    plt.boxplot([ell_data, sv_data, nd_obs_data, nd_data], labels=xlabels)
    ax.set_ylabel('Max time between target visits')
    plt.title('Max time between visits, %s domain' % env)
    plt.show()

def comboplot_mean_paper(ell_stats, sv_stats, nd_obs_stats, nd_stats):
    env = 'custom2'
    ell_data1 = [s[0]/8. for s in ell_stats[env]]
    sv_data1 = [s[0]/8. for s in sv_stats[env]]
    nd_obs_data1 = [s[0]/8. for s in nd_obs_stats[env]]
    nd_data1 = [s[0]/8. for s in nd_stats[env]]

    env = 'custom1'
    ell_data2 = [s[0]/8. for s in ell_stats[env]]
    sv_data2 = [s[0]/8. for s in sv_stats[env]]
    nd_obs_data2 = [s[0]/8. for s in nd_obs_stats[env]]
    nd_data2 = [s[0]/8. for s in nd_stats[env]]

    xlabels =['Static Ellipsoid Decomposition (ours)', 'Static Voronoi', 'Dynamic Ellipsoid Decomposition (ours)', 'Dynamic Voronoi']

    matplotlib.rcParams.update({'font.size':26})
    fig,ax = plt.subplots()
    xpos = np.array([100, 200, 300, 400])
    c = 'r'
    plt.boxplot([ell_data1, sv_data1, nd_obs_data1, nd_data1], positions=xpos-10, widths=15, boxprops=dict(color=c), capprops=dict(color=c),whiskerprops=dict(color=c), flierprops=dict(markeredgecolor=c))
    plt.boxplot([ell_data2, sv_data2, nd_obs_data2, nd_data2], positions=xpos+10, widths=15)
    plt.plot([],[], color='r', label='Spiral Environment')
    plt.plot([],[], color='k', label='Jagged Environment')
    ax.set_ylabel('Mean time between target visits (s)')
    plt.xticks(xpos)
    ax.set_xticklabels(xlabels)
    plt.title('Mean Time Between Visits')
    plt.legend()
    plt.show()

def comboplot_max_paper(ell_stats, sv_stats, nd_obs_stats, nd_stats):
    env = 'custom2'
    ell_data1 = [s[2]/8. for s in ell_stats[env]]
    sv_data1 = [s[2]/8. for s in sv_stats[env]]
    nd_obs_data1 = [s[2]/8. for s in nd_obs_stats[env]]
    nd_data1 = [s[2]/8. for s in nd_stats[env]]

    env = 'custom1'
    ell_data2 = [s[2]/8. for s in ell_stats[env]]
    sv_data2 = [s[2]/8. for s in sv_stats[env]]
    nd_obs_data2 = [s[2]/8. for s in nd_obs_stats[env]]
    nd_data2 = [s[2]/8. for s in nd_stats[env]]

    xlabels =['Static Ellipsoid Decomposition (ours)', 'Static Voronoi', 'Dynamic Ellipsoid Decomposition (ours)', 'Dynamic Voronoi']

    matplotlib.rcParams.update({'font.size':26})
    fig,ax = plt.subplots()
    xpos = np.array([100, 200, 300, 400])
    c = 'r'
    plt.boxplot([ell_data1, sv_data1, nd_obs_data1, nd_data1], positions=xpos-10, widths=15, boxprops=dict(color=c), capprops=dict(color=c),whiskerprops=dict(color=c), flierprops=dict(markeredgecolor=c))
    plt.boxplot([ell_data2, sv_data2, nd_obs_data2, nd_data2], positions=xpos+10, widths=15)
    plt.plot([],[], color='r', label='Spiral Environment')
    plt.plot([],[], color='k', label='Jagged Environment')
    ax.set_ylabel('Max time between target visits (s)')
    plt.xticks(xpos)
    ax.set_xticklabels(xlabels)
    plt.title('Max Time Between Visits')
    plt.legend()
    plt.show()

   

logdir = 'logs_long_full'
files = os.listdir(logdir)

domain_names = ['blocks', 'forest_', 'custom1', 'custom2']

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


comboplot_mean_paper(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats)
comboplot_max_paper(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats)

plot_max_timing_comparison_paper(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'custom1')
plot_max_timing_comparison_paper(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'custom2')

stop

plot_mean_timing_comparison_paper(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'custom1')
plot_mean_timing_comparison_paper(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'custom2')


#plot_mean_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'blocks')
#plot_mean_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'forest_')
plot_mean_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'custom1')
plot_mean_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'custom2')

#plot_max_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'blocks')
#plot_max_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'forest_')
plot_max_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'custom1')
plot_max_timing_comparison(ellipsoids_stats, static_voronoi_stats, no_decomp_obs_stats, no_decomp_stats, 'custom2')




