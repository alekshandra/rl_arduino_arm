import gym
import numpy as np
from matplotlib import pyplot as plt


class Environment(gym.Env):
    """
    ### Description
    This environment represents a robotic arm with 4 joints: one is at the base allowing the arm to rotate about the z-axis;
    the rest of the joints operate in a plane and the environment makes use of this to calculate positions.

    ### Action Space
    The agent takes a 4-element vector for actions.
    Each element can take a continuous range of values between -1 and 1.
    This is scaled down by a factor and corresponds to how much each angle should change

    ### Observation Space
    The observation is an array with shape (10,) with elements corresponding to:
    | Num | Observation | Min | Max|
    |-----|-------------|-----|----|
    |  0  | Curr Angle0 |  0  | 180|
    |  1  | Curr Angle1 |  0  | 180|
    |  2  | Curr Angle2 |  0  | 180|
    |  3  | Curr Angle3 |  0  | 180|
    |  4  | Curr Sensor |  0  |   1|
    |  5  | Prev Angle0 |  0  | 180|
    |  6  | Prev Angle1 |  0  | 180|
    |  7  | Prev Angle2 |  0  | 180|
    |  8  | Prev Angle3 |  0  | 180|
    |  9  | Prev Sensor |  0  |   1|

    ### Rewards and Penalties
    A reward is given when the arm moves, this is equal to the sum of the absolute difference
    in angles from one state to the next
    The environment penalises the robot if:
        1. the wrist is not pointing downwards, a necessary condition for the light sensor to register anything
        2. it moves off the line, the penalty gets heavier for each step it remains off the line
        3. violating physical constraints - going through the floor, exceeding ranges of angles

    ### Starting State
    The environment is initialised with the angles at [90, 90, 0, 0]
    The arm looks like this:
                                     ============
                                     ||        ||
                                     ||        ||
                                     ||
                              _______||________________

    ### Episode Termination
    The episode terminates when the arm spends 10 or more turns off the line
    """
    def __init__(self):
        self.current_state = None
        # self.previous_previous_state = None
        self.light_z_angle = None
        self.light_planar_positions = None
        self.light_positions = None
        self.planar_positions = None
        self.positions = None
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=np.array([180, 180, 180, 180, 1, 180, 180, 180, 180, 1]),
                                                shape=(4+1+4+1,))

        self.base_height = 75
        self.first_length = 125
        self.second_length = 125
        self.wrist_length = 60
        self.lengths = (self.base_height, self.first_length, self.second_length, self.wrist_length)
        self.angles = np.array([0.0, 0.0, 0.0, 0.0])

        self.calculate_positions()
        self.calculate_light_beam()

        self.shape_coord = np.array([[0, 125],
                                     [75, 125],
                                     [75, 200],
                                     [0, 200],
                                     [0, 125]])

        granular_x_coord = np.linspace(self.shape_coord[:-1, 0], self.shape_coord[1:, 0], 10000).T.ravel()
        granular_y_coord = np.linspace(self.shape_coord[:-1, 1], self.shape_coord[1:, 1], 10000).T.ravel()

        self.granular_coord = np.vstack((granular_x_coord, granular_y_coord)).T
        self.off_line_counter = 0
        self.state = None
        self.previous_state = None

    def reset(self):
        angle1 = 90.0
        angle2 = 90.0
        angle3 = 0.0
        angle4 = 0.0

        self.angles = np.array([angle1, angle2, angle3, angle4], dtype=np.float32)
        self.calculate_positions()
        self.calculate_light_beam()
        close_x = np.abs(self.light_positions[1, 0]-self.granular_coord[:, 0]) < 1
        close_y = np.abs(self.light_positions[1, 1]-self.granular_coord[:, 1]) < 1
        # close = np.isclose(self.light_positions[1, :-1], self.granular_coord, rtol=0, atol=1)
        close = np.any(np.logical_and(close_x, close_y))
        self.current_state = np.concatenate([self.angles, [close]])
        self.previous_state = self.current_state
        self.state = np.concatenate([self.current_state, self.previous_state])
        self.off_line_counter = 0
        # self.previous_previous_state = None

        return self.state

    def step(self, action):
        print("Environment State: ", self.state)
        # self.previous_previous_state = self.previous_state
        self.previous_state = self.current_state
        self.angles += action/100

        self.calculate_positions()
        self.calculate_light_beam()

        print("Environment Light Positions: ", self.light_positions[1, :-1])
        close_x = np.abs(self.light_positions[1, 0] - self.granular_coord[:, 0]) < 1
        close_y = np.abs(self.light_positions[1, 1] - self.granular_coord[:, 1]) < 1
        # close = np.isclose(self.light_positions[1, :-1], self.granular_coord, rtol=0, atol=1)
        close = np.any(np.logical_and(close_x, close_y))
        print("Environment Close: ", close)
        self.current_state = np.concatenate([self.angles, [close]])
        self.state = np.concatenate([self.current_state, self.previous_state])

        reward = 0
        print("Environment State: ", self.state)
        if not close:
            self.off_line_counter += 1
            print("Environment Off Line Counter: ", self.off_line_counter)
        if close and self.off_line_counter > 0:
            self.off_line_counter = 0
            reward += 100 * self.off_line_counter ** 2
            print("Environment Off Line Counter (RESET!): ", self.off_line_counter)

        done = self.off_line_counter >= 10

        print("Environment Wrist Penalty: ", -100 * np.abs(self.light_z_angle + np.pi/2 + 0.000001))
        print("Environment Off Line Penalty: ", 10 * self.off_line_counter ** 2)
        print("Environment Angle Change Reward: ", 100 * np.sum(np.abs(self.current_state[:4] - self.previous_state[:4])))
        reward -= 100 * np.abs(self.light_z_angle + np.pi/2 + 0.000001)  # Penalty for the wrist not pointing down
        reward -= 10 * self.off_line_counter ** 2  # Penalty for being off the line, will increase with time spent off
        reward += 100 * np.sum(np.abs(self.current_state[:4] - self.previous_state[:4]))  # Reward for changing angles

        # if self.previous_previous_state is not None:
        #     reward += np.sum(np.abs(self.current_state - self.previous_previous_state)) Reward for changing the angles
        if self.positions[-1, -1] < 0:
            reward -= 1000000  # Penalty for going below the ground
            done = True
        if self.angles[0] < 0 or self.angles[0] > 180:
            reward -= 1000000  # Penalty for violating angle 0 constraint
            done = True
        if self.angles[1] < 0 or self.angles[1] > 180:
            reward -= 1000000  # Penalty for violating angle 1 constraint
            done = True
        if self.angles[2] < 0 or self.angles[2] > 180:
            reward -= 1000000  # Penalty for violating angle 2 constraint
            done = True
        if self.angles[3] < 0 or self.angles[3] > 180:
            reward -= 1000000  # Penalty for violating angle 3 constraint
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        ax = plt.axes(projection='3d')
        ax.plot3D(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], 'g')
        ax.plot3D(self.light_positions[:, 0], self.light_positions[:, 1], self.light_positions[:, 2], 'r')
        ax.plot3D(self.shape_coord[:, 0], self.shape_coord[:, 1], np.zeros(self.shape_coord.shape[0]), 'b')
        plt.show()

    def calculate_positions(self):
        """
        Calculates the positions of each joint and the reflection of the light sensor's beam
        The arm operates in a plane so positions are calculated in 2D frame of reference then converted to 3D
        """
        angles = np.radians(self.angles)
        joint1_planar_pos = np.array([0, 0])
        joint2_planar_pos = np.array([0, self.lengths[0]])
        joint3_planar_pos = joint2_planar_pos + self.lengths[1] * np.array([np.cos(angles[1]), np.sin(angles[1])])

        joint4_planar_pos = joint3_planar_pos + self.lengths[2] * np.array([np.cos(angles[1] + angles[2] - np.pi/2),
                                                                            np.sin(angles[1] + angles[2] - np.pi/2)])

        joint5_planar_pos = joint4_planar_pos + self.lengths[3] * np.array([np.cos(angles[1] + angles[2] +
                                                                                   angles[3]-np.pi),
                                                                            np.sin(angles[1] + angles[2] +
                                                                                   angles[3]-np.pi)])

        self.planar_positions = np.array([joint1_planar_pos, joint2_planar_pos, joint3_planar_pos,
                                          joint4_planar_pos, joint5_planar_pos])

        self.positions = np.zeros((5, 3))
        self.positions[:, 2] = self.planar_positions[:, 1]
        self.positions[:, 0] = self.planar_positions[:, 0]*np.cos(angles[0])
        self.positions[:, 1] = self.planar_positions[:, 0]*np.sin(angles[0])

    def calculate_light_beam(self):
        """
        Calculates the trajectory of the light sensor's beam
        The arm operates in a plane so positions are calculated in 2D frame of reference then converted to 3D
        """
        angles = np.radians(self.angles)
        joint5_planar_pos = self.planar_positions[-1, :]
        self.light_z_angle = angles[1] + angles[2] + angles[3] - np.pi
        light_check = self.light_z_angle < 0 or self.light_z_angle > np.pi
        if light_check:
            light_delta = joint5_planar_pos[1] * np.tan(2.5 * np.pi - angles[1] - angles[2] - angles[3])
            light_hits_ground = np.array([joint5_planar_pos[0] - light_delta, 0])
            light_comes_back = joint5_planar_pos - 2 * np.array([light_delta, 0])
            self.light_planar_positions = np.array([joint5_planar_pos, light_hits_ground, light_comes_back])
        else:
            light_goes_off = joint5_planar_pos.copy()
            light_goes_off += 50 * np.array([np.cos(angles[1] + angles[2] + angles[3] - np.pi),
                                             np.sin(angles[1] + angles[2] + angles[3] - np.pi)])
            self.light_planar_positions = np.array([joint5_planar_pos, light_goes_off])

        self.light_positions = np.zeros((self.light_planar_positions.shape[0], 3))
        self.light_positions[:, 2] = self.light_planar_positions[:, 1]
        self.light_positions[:, 0] = self.light_planar_positions[:, 0] * np.cos(angles[0])
        self.light_positions[:, 1] = self.light_planar_positions[:, 0] * np.sin(angles[0])
