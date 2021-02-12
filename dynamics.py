#!/usr/bin/python3

import numpy as np
import casadi as cd


def update_agents(agents, dt):
    for t in agents.agent_list:
        state_new = step(t.unicycle_state, t.control, t.params, dt, mpcc=True)
        t.unicycle_state[:] = state_new

def step(x, u, p, dt, mpcc=False):
    
    if mpcc:
        f = mpcc_ddt
    else:
        f = unicycle_ddt

    k1 = f(x, u, p)
    k2 = f(x + dt * k1/2., u, p)
    k3 = f(x + dt * k2/2., u, p)
    k4 = f(x + dt * k3, u, p)

    x_new = x + (k1 + 2*k2 + 2*k3 + k4) * dt/6.

    return x_new

def mpcc_ddt(x, u, p, casadi=False):
    x_tracker = x[:-1]
    u_tracker = u[:-1]
    ddt_tracker = unicycle_ddt(x_tracker, u_tracker, p, casadi=casadi)
    if casadi:
        return cd.vertcat(ddt_tracker, u[-1])
    else:
        ddt = np.zeros(6, dtype=float)
        ddt[:-1] = ddt_tracker
        ddt[-1] = u[-1]
        return ddt


def unicycle_ddt(x, u, p, casadi=False):

    if casadi:
        cos = cd.cos
        sin = cd.sin
    else:
        cos = np.cos
        sin = np.sin

    position = x[:2]
    theta = x[2:3]
    velocity = x[3:4]
    omega = x[4:5]


    # body frame
    #linear_acceleration = u[:3]
    #angular_acceleration = u[3:6]

    a_x = u[0:1]
    alpha = u[1:2]

    ddt_vx = a_x * cos(theta) - velocity * sin(theta) * omega
    ddt_vy = a_x * sin(theta) + velocity * cos(theta) * omega
   
    if casadi:
        ddt = np.zeros(5, dtype=object)
    else: 
        ddt = np.zeros(5, dtype=float)
    # field remapping
    ddt_position = ddt[:2]
    ddt_orientation = ddt[2:3]
    ddt_velocity = ddt[3:4]
    ddt_angular_velocity = ddt[4:5]


    ddt_x = velocity * cos(theta)
    ddt_y = velocity * sin(theta)
    ddt_position[0] = ddt_x
    ddt_position[1] = ddt_y
    ddt_orientation[0] = omega

    ddt_velocity[0] = a_x

    ddt_angular_velocity[0] = alpha

    if not casadi:
        if velocity < p.min_velocity:
            ddt_velocity[0] = np.max([0, ddt_velocity[0]])
        if velocity > p.max_velocity:
            ddt_velocity[0] = np.min([0, ddt_velocity[0]])
        #ddt_position[:] = np.clip(ddt_position, p.min_velocity, p.max_velocity)
        ddt_orientation[:] = np.clip(ddt_orientation, p.min_angular_velocity, p.max_angular_velocity)
    else:
        ddt = cd.vertcat(*ddt)

    return ddt

