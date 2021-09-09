#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
import pickle
import sys

print('Launch as `python3 env_designer.py <output_pickle_path> \n Click to define the convex hull of a polygon.\n Press spacebar to start a new polygon. \nPress delete to undo a clicked point.\nPress s to save the obstacles (and ignore the pyplot save box that appears)')

output_filename = sys.argv[1]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([-15, 15])
plt.ylim([-15, 15])

coords = []
cur_coords = []
polys = []
cur_poly = patches.Polygon([[0, 0],[1,0], [0,1]], True)
ax.add_patch(cur_poly)

def onclick(event):
    global cur_poly
    ix, iy = event.xdata, event.ydata
    cur_coords.append((ix, iy))

    if len(cur_coords) > 2:
        hull = ConvexHull(np.array(cur_coords))
        hull_pts = np.array(cur_coords)[hull.vertices]
        print(hull_pts)
        cur_poly.set_xy(hull_pts)
        fig.canvas.draw()
    

def onkeypress(event):
    global polys, cur_poly, cur_coords, coords
    print(event.key)
    if event.key == ' ':
        print('next polygon')
        coords.append(np.array(cur_coords))
        polys.append(cur_poly)
        cur_poly = patches.Polygon([[np.nan,np.nan]], True)
        cur_coords = []
        ax.add_patch(cur_poly)
    if event.key == 'delete':
        cur_coords = cur_coords[:-1]
        if len(cur_coords) > 2:
            hull = ConvexHull(np.array(cur_coords))
            hull_pts = np.array(cur_coords)[hull.vertices]
            print(hull_pts)
            cur_poly.set_xy(hull_pts)
            fig.canvas.draw()
    if event.key == 's':
        with open(output_filename, 'wb') as fo:
            pickle.dump(coords, fo)
        print('saved to %s' % output_filename)
        
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('key_press_event', onkeypress)

plt.show()
