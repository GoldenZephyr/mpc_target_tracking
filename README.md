"Backback Problem" Tracking
===========================


scratch todo
------------

[x] prediction
[x] distance
[x] TSP high level plan
    [x] write distance matrix to file
    [x] call pytsp
    [x] draw tsp
[x] shot order ILP
    [x] roll out predictions
    [x] uniform random sampling of times
    [x] build ilp -- clip based on expected distance
    [x] solve + unroll assignment
    [x] redo/reorder dummy variables in DAG
    [x] put the pieces together
    [x] demo with static agents
    [x] plot test case
[x] Video baselines
    [x] TSP video
    [x] ILP video
    [x] ILP with KNN limitation
[x] multi-tracker (naive division)
[x] polish so it works all the way through
[ ] record metrics about how long it takes
[ ] simple additions to reduce assignment instability

At that point, have a simple proof of concept of the full stack.
will need to:
1) improve high level assignment
2) investigate swaps
3) port to drone sim (?)

Implementation Phases
=====================


Initial Functionality
---------------------

The first part of the implementation is a simple environment baseline. 

[x] simple high level plan
[x] mpc terminal cost
[x] target prediction
[x] distance constraint
[ ] casadi Map
[ ] code cleanup / function docs


Solve the Tricky Pieces
-----------------------

This is where the kernel of innovation is. We need to constrain the tracker to
enter the desired regions, and implement a smart high level plan.

[x] viewpoint constraint
    * compare hard constraint vs "lyapunov" version
    * consider alyssa's point about optimal boundary position
[x] real high level plan



Multiagent Functionality
------------------------

[x] Dividing agent assignment
[ ] swapping agent assignment


Domain Functionality
--------------------

We will also include support for obstacles in the environment. However, we will
only support ellipsoidal obstacles. More complicated obstacles will be relegated
to future work because they are too different from the existing codebase.

[ ] Obstacles


Julia Port
----------

It would be nice to port all of this to Julia to aide functionality. Julia seems
to have very good python support (and the MPC definition could stay in python
directly anyway), so it's just the rest of the sim that needs to be translated.



