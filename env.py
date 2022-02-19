import gym
import numpy as np
from matplotlib import pyplot as plt


class Environment(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.array([180, 180, 180, 180, 1]), shape=(4+1,))

        self.base_height = 75
        self.first_length = 125
        self.second_length = 125
        self.wrist_length = 60
        self.lengths = (self.base_height, self.first_length, self.second_length, self.wrist_length)
        self.angles = np.array([0.0, 0.0, 0.0, 0.0])

        self.calculate_positions()
        self.calculate_light_beam()

        self.shape_coordinates = np.array([[0, 125],
                                           [75, 125],
                                           [75, 200],
                                           [0, 200],
                                           [0, 125]])

        granular_x_coordinates = np.linspace(self.shape_coordinates[:-1, 0], self.shape_coordinates[1:, 0],
                                             10000).T.ravel()
        granular_y_coordinates = np.linspace(self.shape_coordinates[:-1, 1], self.shape_coordinates[1:, 1],
                                             10000).T.ravel()

        self.granular_coordinates = np.vstack((granular_x_coordinates, granular_y_coordinates)).T

    def reset(self):
        angle1 = 90.0
        angle2 = 90.0
        angle3 = 0.0
        angle4 = 0.0

        self.angles = np.array([angle1, angle2, angle3, angle4], dtype=np.float32)
        self.calculate_positions()
        self.calculate_light_beam()

        self.state = np.ones(5, dtype=np.float32)
        self.state[:4] = self.angles

        return self.state

    def step(self, action):
        new_angles = self.angles.copy()
        # print(new_angles[0], action[0])
        # print(new_angles[0] + action[0])
        new_angles[0] += action[0]
        # print(new_angles[0])
        # new_angles[1] = self.angles[1]  + action[1]
        # new_angles[2] = self.angles[2]  + action[2]
        # new_angles[3] = self.angles[3]  + action[3]

        self.angles = new_angles

        self.calculate_positions()
        self.calculate_light_beam()

        self.state = np.ones(5, dtype=np.float32)
        self.state[:4] = new_angles

        # print(state)
        # close = np.isclose(x, self.granular_coordinates)
        # np.any(np.logical_and(close[:, 0], close[:, 1]))

        done = np.isclose(new_angles[0], 180.0, 0.01)  # or new_angles[0] == 0

        reward = -1 * np.abs(new_angles[0] - 180)
        # if done:
        #     reward = 0
        # else:
        #     reward = -1

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        ax = plt.axes(projection='3d')
        ax.plot3D(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], 'g')
        ax.plot3D(self.light_positions[:, 0], self.light_positions[:, 1], self.light_positions[:, 2], 'r')
        ax.plot3D(self.shape_coordinates[:, 0], self.shape_coordinates[:, 1], np.zeros(self.shape_coordinates.shape[0]), 'b')
        plt.show()

    def calculate_positions(self):
        """
        Calculates the positions of each joint and the reflection of the light sensor's beam
        The arm operates in a plane so positions are calculated in 2D frame of reference then converted to 3D
        """
        angles = np.radians(self.angles)
        joint1_planar_pos = np.array([0, 0])
        joint2_planar_pos = np.array([0, self.lengths[0]])
        joint3_planar_pos = joint2_planar_pos + self.lengths[1] * np.array([np.cos(angles[1]),
                                                                            np.sin(angles[1])])

        joint4_planar_pos = joint3_planar_pos + self.lengths[2] * np.array([np.cos(angles[1] +
                                                                                   angles[2] - np.pi/2),
                                                                            np.sin(angles[1] +
                                                                                   angles[2] - np.pi/2)])

        joint5_planar_pos = joint4_planar_pos + self.lengths[3] * np.array([np.cos(angles[1] +
                                                                                   angles[2] +
                                                                                   angles[3] - np.pi),
                                                                            np.sin(angles[1] +
                                                                                   angles[2] +
                                                                                   angles[3] - np.pi)])

        self.planar_positions = np.array([joint1_planar_pos, joint2_planar_pos, joint3_planar_pos,
                                          joint4_planar_pos, joint5_planar_pos])

        self.positions = np.zeros((5, 3))
        self.positions[:, 2] = self.planar_positions[:, 1]
        self.positions[:, 0] = self.planar_positions[:, 0] * np.cos(angles[0])
        self.positions[:, 1] = self.planar_positions[:, 0] * np.sin(angles[0])

    def calculate_light_beam(self):
        """
        Calculates the trajectory of the light sensor's beam
        The arm operates in a plane so positions are calculated in 2D frame of reference then converted to 3D
        """
        angles = np.radians(self.angles)
        joint5_planar_pos = self.planar_positions[-1, :]
        self.light_z_angle = angles[1]+angles[2]+angles[3]-np.pi
        light_check = self.light_z_angle < 0 or self.light_z_angle > np.pi
        if light_check:
            light_delta = joint5_planar_pos[1]*np.tan(2.5*np.pi-angles[1]-angles[2]-angles[3])
            lightbeam_hits_ground = np.array([joint5_planar_pos[0]-light_delta, 0])
            lightbeam_comes_back = joint5_planar_pos-2*np.array([light_delta, 0])
            self.light_planar_positions = np.array([joint5_planar_pos, lightbeam_hits_ground, lightbeam_comes_back])
        else:
            lightbeam_goes_off = joint5_planar_pos.copy()
            lightbeam_goes_off += 50*np.array([np.cos(angles[1]+angles[2]+angles[3]-np.pi),
                                               np.sin(angles[1]+angles[2]+angles[3]-np.pi)])
            self.light_planar_positions = np.array([joint5_planar_pos, lightbeam_goes_off])

        self.light_positions = np.zeros((self.light_planar_positions.shape[0], 3))
        self.light_positions[:, 2] = self.light_planar_positions[:, 1]
        self.light_positions[:, 0] = self.light_planar_positions[:, 0]*np.cos(angles[0])
        self.light_positions[:, 1] = self.light_planar_positions[:, 0]*np.sin(angles[0])

