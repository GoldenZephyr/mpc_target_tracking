# agents.py -- Class for holding the agents

import numpy as np


class DefaultTrackerParams:
    def __init__(self):
        self.min_velocity = np.array([0])
        self.max_velocity = np.array([3])
        self.min_angular_velocity = np.array([-3])
        self.max_angular_velocity = np.array([3])
        self.switch_ix = 39
        self.ellipse_switch_ix = 40
        self.ellipse_switch_pair = [-1, -1]

class DefaultTargetParams:
    def __init__(self):
        self.min_velocity = np.array([0])
        self.max_velocity = np.array([1.5])
        self.min_angular_velocity = np.array([-np.pi/15.0])
        self.max_angular_velocity = np.array([np.pi/15.0])
        self.information_angle = 0.0
        self.information_tolerance = np.radians(30)

class Agent:
    def __init__(self, state, unicycle_state, control, agent_params, ix):
        self.index = ix
        self.state = state
        self.unicycle_state = unicycle_state
        self.unicycle_state_mpc = self.unicycle_state[:5]

        self.position = self.state[:3]
        self.orientation = self.state[3:6]
        self.velocity = self.state[6:9]
        self.angular_velocity = self.state[9:12]


        self.control = control
        self.linear_acceleration = self.control[0:1]
        self.angular_acceleration = self.control[1:2]
        self.mpc_control = self.control[:2]

        self.params = agent_params


class AgentGroup:
    def __init__(self, n, min_lim, max_lim, agent_params):


        # 12 DOF state (x,y,z,r,p,y, vx, vy, vz, dr, dp, dy)
        self.state = np.zeros((n, 12))


        # 6 DOF pose for each agent.
        self.pose = self.state[:,:6]
        self.pose[:,:3] = np.random.uniform(min_lim, max_lim, size=(n, 3))
        self.pose[:,5] = np.random.uniform(-np.pi, np.pi, size=n)

        # position and orientation broken out for convenience, reference same data as self.pose
        self.position = self.pose[:,:3]
        self.orientation = self.pose[:,3:]

        # Linear Velocity
        self.velocity = self.state[:, 6:9]

        # Angular Velocity
        self.angular_velocity = self.state[:, 9:12]

        # Control inputs (body frame)
        #self.control = np.zeros((n, 6))
        #self.linear_acceleration = self.control[:,:3]
        #self.angular_acceleration = self.control[:,3:6]


        # 5 DOF unicycle state (x, y, theta, vel, omega, mpcc_theta)
        self.unicycle_state = np.zeros((n, 6))

        self.unicycle_position = self.unicycle_state[:,:2]
        self.unicycle_orientation = self.unicycle_state[:,2:3]
        self.unicycle_velocity = self.unicycle_state[:,3:4]
        self.unicycle_angular_velocity = self.unicycle_state[:,4:5]
        self.unicycle_mpcc_theta = self.unicycle_state[5:6]

        self.unicycle_position[:] = self.position[:,:2]
        self.unicycle_orientation[:] = self.orientation[:,2:3]

        # 3 DOF unicycle control (a_x, \alpha, mpcc_theta_dot)
        self.unicycle_control = np.zeros((n,3))
        self.unicycle_acceleration = self.unicycle_control[:,0:1]
        self.unicycle_angular_acceleration = self.unicycle_control[:,1:2]
        self.unicycle_mpcc_theta_dot = self.unicycle_control[:,2:3]
        

        # Construct list of individual agent objects that share the same underlying representation
        self.agent_list = [0] * n
        for ix in range(n):
            self.agent_list[ix] = Agent(self.state[ix,:], self.unicycle_state[ix, :], self.unicycle_control[ix,:], agent_params, ix)

    def synchronize_state(self):
        """ Updates the 12 DOF state to reflect the unicycle state
        """
        theta = self.unicycle_state[:,2]
        v = self.unicycle_state[:,3]
        self.position[:, :2] = self.unicycle_state[:,:2]
        self.orientation[:,2] = theta
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)

        self.velocity[:, 0] = vx
        self.velocity[:, 1] = vy

        self.angular_velocity[:, 2] = self.unicycle_state[:, 4]



