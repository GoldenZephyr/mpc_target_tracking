#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os


keys_trackers = [1,2,5,7]
keys_targets = [5,10,20,50]
keys_domains = ['blocks', 'forest']

combos = [(ktrack, ktarget, kdomain) for ktrack in keys_trackers for ktarget in keys_targets for kdomain in keys_domains]
sim_length_dict = {}
tsp_time_dict = {}
mpc_time_dict = {}
tsp_std_dict = {}
mpc_std_dict = {}
for k in combos:
    sim_length_dict[k] = []
    tsp_time_dict[k] = []
    mpc_time_dict[k] = []
    tsp_std_dict[k] = []
    mpc_std_dict[k] = []

files = os.listdir('data_out')

failed_files = []
for fn in files:
    toks = fn.split('_')
    domain = toks[1]
    n_trackers = int(toks[3])
    n_targets = int(toks[5])
    gen = int(toks[7])
    with open('data_out/' + fn, 'r') as fo:
        lines = fo.readlines()
    if len(lines) < 2:
        failed_files.append(fn)
        continue
    n_steps = int(lines[1])
    tsp_time = float(lines[2].split(',')[0])
    tsp_std = float(lines[2].split(',')[1])
    mpc_time = float(lines[3].split(',')[0])
    mpc_std = float(lines[3].split(',')[1])


    key = (n_trackers, n_targets, domain)
    sim_length_dict[key].append(n_steps)
    tsp_time_dict[key].append(tsp_time)
    mpc_time_dict[key].append(mpc_time)
    tsp_std_dict[key].append(tsp_std)
    mpc_std_dict[key].append(mpc_std)


print('%d files failed to parse' % len(failed_files))
print(failed_files)

simtime_v_targets_dict = {}
simtime_v_trackers_dict = {}
tsptime_v_targets = plt.subplots(1,1)
tsptime_v_targets[1].set_title('TSP Solution time vs. # Target')
tsptime_v_targets[1].set_xlabel('# Targets')
tsptime_v_targets[1].set_ylabel('Time (s)')

for k in keys_trackers:
    for d in keys_domains:
        fig, ax = plt.subplots(1,1)
        simtime_v_targets_dict[(d,k)] = (fig, ax)
        ax.set_title('Time vs. # Targets, %d Trackers, %s' % (k, d))
        ax.set_xlabel('# Targets')
        ax.set_ylabel('Time (steps)')

for k in keys_targets:
    for d in keys_domains:
        fig, ax = plt.subplots(1,1)
        simtime_v_trackers_dict[(d,k)] = (fig, ax)
        ax.set_title('Time vs. # Trackers, %d Targets, %s' % (k, d))
        ax.set_xlabel('# Trackers')
        ax.set_ylabel('Time (steps)')

for k in combos:
    ntrack, ntarget, domain = k
    
    n = len(sim_length_dict[k])
    _, ax1 = simtime_v_targets_dict[(domain, ntrack)]
    ax1.scatter([ntarget]*n, sim_length_dict[k])

    _, ax2 = simtime_v_trackers_dict[(domain, ntarget)]
    ax2.scatter([ntrack]*n, sim_length_dict[k])

    if ntrack == 1:
        _, ax3 = tsptime_v_targets
        ax3.scatter([ntarget]*n, tsp_time_dict[k])


for k in keys_trackers:
    for d in keys_domains:
        fig, _ = simtime_v_targets_dict[(d,k)]
        fig.savefig('plots/simtime_v_targets_%d_trackers_%s.png' % (k, d))

for k in keys_targets:
    for d in keys_domains:
        fig, _ = simtime_v_trackers_dict[(d,k)]
        fig.savefig('plots/simtime_v_trackers_%d_targets_%s.png' % (k, d))

fig, _ = tsptime_v_targets
fig.savefig('plots/tsptime.png')
    

print(np.mean(sim_length_dict[(1,10,'blocks')]))
print(np.std(sim_length_dict[(1,10,'blocks')]))
print('----')
print(np.mean(sim_length_dict[(1,20,'blocks')]))
print(np.std(sim_length_dict[(1,20,'blocks')]))
print('----')
print(np.mean(sim_length_dict[(2,10,'blocks')]))
print(np.std(sim_length_dict[(2,10,'blocks')]))
print('----')
print(np.mean(sim_length_dict[(3,10,'blocks')]))
print(np.std(sim_length_dict[(3,10,'blocks')]))
