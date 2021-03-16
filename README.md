"Backback Problem" Tracking
===========================


To run the script, first run `python3 mpc_iris_2.py` to generate the MPC. Then,
run `python3 run_obstacles.py`. This should generate an output video in the
`videos` subdirectory. You can modify then number of targets and trackers in
`run_obstacles.py`. 

In general, I would re-implement the general MPC idea from here into your
existing control stack. The core ideas are quite simple, and there are just a
few functions that encapsulate the interesting parts. 

`mpc_obs_functions.py`
======================

This file has most of the relevant utility functions.

`construct_ellipse_space`
-------------------------

This takes an environment and decomposes the free space into ellipsoids. It
makes a finely spaced lattice filling the space, and uses the IRIS library to
generate ellipsoids such that all of the lattice points are covered.

`construct_ellipse_topology`
---------------------------

After decomposing the free space into ellipsoids, store the connectivity of
which ellipsoids intersect with each other.

`find_ellipsoid_path`
---------------------

Given the ellipsoid graph and a desired start and endpoint, finds the fewest
number of ellipsoids to traverse from the start to end. Also generates a
waypoint in the intersection of each pair of ellipsoids on the path.


`utils.py`
==========

`fine_ellipse_intersection`
---------------------------
Given two ellipsoids, find a point in their intersection.

`ellipsoids_intersect`
----------------------
Checking condition for whether two ellipsoids intersect.

`mpc_iris_2.py`
===============

Contains the MPC definition. Uses collocation method to enforce dynamics. You
will see there are weights that specify which indices have a penalty for the
primary goal and which have weights for the secondary goal. These are updated
between iterations in `run_obstacles.py` (`weights_1` and `weights_2` and
`switch_ix`). The active ellipsoid constraint is controlled by the ellipsoid
shape matrices and centers A,B,a,b, and the switching between the two
constraints is defined by the solver bounds `ubg` which are also updated between
runs in `run_obstacles.py`. 
