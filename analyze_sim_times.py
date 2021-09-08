#!/usr/bin/python3

import numpy as np
import os

def analyze_times(fn):
    data = np.genfromtxt(fn, delimiter=',')
    meantime = np.mean(data)
    meansqtime = np.mean(data**2)
    maxtime = np.max(data)
    return meantime, meansqtime, maxtime

# blocks_ellipsoids_4_trackers_15_targets_1631074144.488345.txt
# blocks_no_decomposition_obsaware_4_trackers_15_targets_1631074690.021977.txt

files = os.listdir('logs')

ellipsoids_files = [os.path.join('logs', f) for f in files if 'ellipsoids' in f]
no_decomp_obs = [os.path.join('logs', f) for f in files if 'no_decomposition_obsaware' in f]
no_decomp = [os.path.join('logs', f) for f in files if 'no_decomposition' in f and 'obsaware' not in f]


ellipsoids_stats = [analyze_times(f) for f in ellipsoids_files]
no_decomp_obs_stats = [analyze_times(f) for f in no_decomp_obs]
no_decomp_stats = [analyze_times(f) for f in no_decomp]

ellipsoids_meantime_mean = np.mean([s[0] for s in ellipsoids_stats])
ellipsoids_meantime_std = np.std([s[0] for s in ellipsoids_stats])
ellipsoids_meansqtime_mean = np.mean([s[1] for s in ellipsoids_stats])
ellipsoids_meansqtime_std = np.std([s[1] for s in ellipsoids_stats])
ellipsoids_maxtime_mean = np.mean([s[2] for s in ellipsoids_stats])
ellipsoids_maxtime_std = np.std([s[2] for s in ellipsoids_stats])

print('Ellipsoids meantime: %f +/- %f (1std)' % (ellipsoids_meantime_mean, ellipsoids_meantime_std))
print('Ellipsoids meansqtime: %f +/- %f (1std)' % (ellipsoids_meansqtime_mean, ellipsoids_meansqtime_std))
print('Ellipsoids maxtime: %f +/- %f (1std)' % (ellipsoids_maxtime_mean, ellipsoids_maxtime_std))
print('')


no_decomp_obs_meantime_mean = np.mean([s[0] for s in no_decomp_obs_stats])
no_decomp_obs_meantime_std = np.std([s[0] for s in no_decomp_obs_stats])
no_decomp_obs_meansqtime_mean = np.mean([s[1] for s in no_decomp_obs_stats])
no_decomp_obs_meansqtime_std = np.std([s[1] for s in no_decomp_obs_stats])
no_decomp_obs_maxtime_mean = np.mean([s[2] for s in no_decomp_obs_stats])
no_decomp_obs_maxtime_std = np.std([s[2] for s in no_decomp_obs_stats])

print('no_decomp_obs meantime: %f +/- %f (1std)' % (no_decomp_obs_meantime_mean, no_decomp_obs_meantime_std))
print('no_decomp_obs meansqtime: %f +/- %f (1std)' % (no_decomp_obs_meansqtime_mean, no_decomp_obs_meansqtime_std))
print('no_decomp_obs maxtime: %f +/- %f (1std)' % (no_decomp_obs_maxtime_mean, no_decomp_obs_maxtime_std))
print('')

no_decomp_meantime_mean = np.mean([s[0] for s in no_decomp_stats])
no_decomp_meantime_std = np.std([s[0] for s in no_decomp_stats])
no_decomp_meansqtime_mean = np.mean([s[1] for s in no_decomp_stats])
no_decomp_meansqtime_std = np.std([s[1] for s in no_decomp_stats])
no_decomp_maxtime_mean = np.mean([s[2] for s in no_decomp_stats])
no_decomp_maxtime_std = np.std([s[2] for s in no_decomp_stats])

print('no_decomp_meantime: %f +/- %f (1std)' % (no_decomp_meantime_mean, no_decomp_meantime_std))
print('no_decomp_meansqtime: %f +/- %f (1std)' % (no_decomp_meansqtime_mean, no_decomp_meansqtime_std))
print('no_decomp_maxtime: %f +/- %f (1std)' % (no_decomp_maxtime_mean, no_decomp_obs_maxtime_std))
print('')
