#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os


keys_trackers = [1,2,5,7]
keys_targets = [5,10,20,50]
keys_domains = ['blocks', 'forest']
keys_generation = [1,2,3,4,5,6,7,8]
combos = [(kgen, ktrack, ktarget, kdomain) for ktrack in keys_trackers for ktarget in keys_targets for kdomain in keys_domains for kgen in keys_generation]
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


    key = (gen, n_trackers, n_targets, domain)
    sim_length_dict[key].append(n_steps)
    tsp_time_dict[key].append(tsp_time)
    mpc_time_dict[key].append(mpc_time)
    tsp_std_dict[key].append(tsp_std)
    mpc_std_dict[key].append(mpc_std)


print('%d files failed to parse' % len(failed_files))
print(failed_files)

#print(np.mean(mpc_time_dict[(8, 1, 20, 'blocks')]))
print('blocks, mean:')
print(np.mean(sim_length_dict[(7, 1, 20, 'blocks')]))
print(np.mean(sim_length_dict[(8, 1, 20, 'blocks')]))
print('blocks, std:')
print(np.std(sim_length_dict[(7, 1, 20, 'blocks')]))
print(np.std(sim_length_dict[(8, 1, 20, 'blocks')]))
print('')
print('forest, mean:')
print(np.mean(sim_length_dict[(7, 1, 20, 'forest')]))
print(np.mean(sim_length_dict[(8, 1, 20, 'forest')]))
print('forest, std:')
print(np.std(sim_length_dict[(7, 1, 20, 'forest')]))
print(np.std(sim_length_dict[(8, 1, 20, 'forest')]))

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
    gen, ntrack, ntarget, domain = k
    if gen != 1:
        continue
    
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


fig, ax = plt.subplots(1,1)
ax.set_title('Gen 3 Time vs. Trackers, 20 Targets, blocks')
ax.set_xlabel('# Trackers')
ax.set_ylabel('Time (steps)')
scatter_lines_1 = [ [] for i in range(5)]
for k in combos:
    gen, ntrack, ntarget, domain = k
    if ntarget != 20 or domain !='blocks':
        continue
    n = len(sim_length_dict[k])
    if gen == 3:
        #ax.scatter([ntrack]*n, sim_length_dict[k], color='r', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='r')
        scatter_lines_1[gen-1].append(s)
    elif gen == 2:
        #ax.scatter([ntrack]*n, sim_length_dict[k], color='g', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='g')
        scatter_lines_1[gen-1].append(s)
    elif gen == 1:
        #ax.scatter([ntrack]*n, sim_length_dict[k], color='k', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='k')
        scatter_lines_1[gen-1].append(s)
    elif gen == 6:
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='b')
        scatter_lines_1[3].append(s)
    elif gen == 4:
        #s = ax.scatter([ntrack]*n, sim_length_dict[k], color='b', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='c')
        #s = ax.errorbar([ntrack], np.mean(sim_length_dict[k]), yerr=np.std(sim_length_dict[k]), color='b', fmt='o', alpha=.2)
        scatter_lines_1[4].append(s)

plt.legend((scatter_lines_1[0][0], scatter_lines_1[1][0], scatter_lines_1[2][0], scatter_lines_1[3][0], scatter_lines_1[4][0]), ('TSP + Lookahead', 'NoTSP, No Lookahead', 'No TSP + Lookahead', 'No TSP + Lookahead, No Pred', 'No TSP, lookahead, v2'))
plt.savefig('plots/gen3_simtime_v_trackers_20_targets_blocks.png')

fig, ax = plt.subplots(1,1)
ax.set_title('Gen 3 Time vs. Trackers, 20 Targets, forest')
ax.set_xlabel('# Trackers')
ax.set_ylabel('Time (steps)')
scatter_lines_1 = [ [] for i in range(5)]
for k in combos:
    gen, ntrack, ntarget, domain = k
    if ntarget != 20 or domain !='forest':
        continue
    n = len(sim_length_dict[k])
    if gen == 3:
        #s = ax.scatter([ntrack]*n, sim_length_dict[k], color='r', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='r')
        #s = ax.errorbar([ntrack], np.mean(sim_length_dict[k]), yerr=np.std(sim_length_dict[k]), color='r', fmt='o', alpha=.2)
        scatter_lines_1[gen-1].append(s)
    elif gen == 2:
        #s = ax.scatter([ntrack]*n, sim_length_dict[k], color='g', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='g')
        #s = ax.errorbar([ntrack], np.mean(sim_length_dict[k]), yerr=np.std(sim_length_dict[k]), color='g', fmt='o', alpha=.2)
        scatter_lines_1[gen-1].append(s)
    elif gen == 1:
        #s = ax.scatter([ntrack]*n, sim_length_dict[k], color='k', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='k')
        #s = ax.errorbar([ntrack], np.mean(sim_length_dict[k]), yerr=np.std(sim_length_dict[k]), color='k', fmt='o', alpha=.2)
        scatter_lines_1[gen-1].append(s)
    elif gen == 6:
        #s = ax.scatter([ntrack]*n, sim_length_dict[k], color='b', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='b')
        #s = ax.errorbar([ntrack], np.mean(sim_length_dict[k]), yerr=np.std(sim_length_dict[k]), color='b', fmt='o', alpha=.2)
        scatter_lines_1[3].append(s)
    elif gen == 4:
        #s = ax.scatter([ntrack]*n, sim_length_dict[k], color='b', alpha=0.2)
        s = ax.scatter([ntrack], np.mean(sim_length_dict[k]), color='c')
        #s = ax.errorbar([ntrack], np.mean(sim_length_dict[k]), yerr=np.std(sim_length_dict[k]), color='b', fmt='o', alpha=.2)
        scatter_lines_1[4].append(s)

plt.legend((scatter_lines_1[0][0], scatter_lines_1[1][0], scatter_lines_1[2][0], scatter_lines_1[3][0], scatter_lines_1[4][0]), ('TSP + Lookahead', 'NoTSP, No Lookahead', 'No TSP + Lookahead', 'No TSP + Lookahead, No Pred', 'No TSP, lookahead, v2'))
plt.savefig('plots/gen3_simtime_v_trackers_20_targets_forest.png')
        
plt.figure()
ntracks = [1,2,5,7]
times =  [np.mean(np.array(sim_length_dict[(1, n, 20, 'blocks')])/8.) for n in ntracks]
yerrs =  [np.std(np.array(sim_length_dict[(1, n, 20, 'blocks')])/8.) for n in ntracks]
plt.errorbar(ntracks, times, yerr=yerrs, fmt='o-')
plt.title('Completion Time vs. # Trackers, 20 targets, Urban Environment')
plt.xlabel('N trackers')
plt.ylabel('Simulation Time (s)')
plt.savefig('plots/blocks_time_paper.png')

plt.figure()
ntracks = [1,2,5,7]
times =  [np.mean(np.array(sim_length_dict[(1, n, 20, 'forest')])/8.) for n in ntracks]
yerrs =  [np.std(np.array(sim_length_dict[(1, n, 20, 'forest')])/8.) for n in ntracks]
plt.errorbar(ntracks, times, yerr=yerrs, fmt='o-')
plt.title('Completion Time vs. # Trackers, 20 targets, Forest Environment')
plt.xlabel('N trackers')
plt.ylabel('Simulation Time (s)')
plt.savefig('plots/forest_time_paper.png')


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
