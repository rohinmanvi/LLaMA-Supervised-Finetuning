import numpy as np
import gym
from gym import spaces
from collections import namedtuple
import math
import matplotlib.pyplot as plt

Action = namedtuple('Action', ['acceleration', 'steering_rate'])

class Vehicle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, phi=0.0, t=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.phi = phi
        self.t = t

class DriverEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.max_steps = 100

        self.episode_r_angle = 0
        self.episode_r_distance = 0

        self.ego_positions = []
        self.agent_positions = []

        self.reset()

        max_v = 15.6464
        max_phi = 0.437
        max_steering_rate = 0.874
        min_a = -10.0
        max_a = 6.0
        max_d = 50
        max_angle = np.pi

        self.action_space = spaces.Box(
            low = np.array([min_a, -max_steering_rate]), 
            high = np.array([max_a, max_steering_rate]), 
            shape = (2,),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low = np.array([0.0, -max_phi, -max_angle, -max_d, -max_angle, -max_v]), 
            high = np.array([max_v, max_phi, max_angle, max_d, max_angle, max_v]), 
            shape = (6,),
            dtype = np.float32
        )

    def step(self, action):
        action = self._map_action_index_to_obj(action)
        self.state, observation, reward = self._generate(self.state, action)

        self.ego_positions.append((self.state[0].x, self.state[0].y))
        self.agent_positions.append((self.state[1].x, self.state[1].y))

        self.steps += 1
        done = self.steps >= self.max_steps

        if done:
            print(f"angle_r: {round(self.episode_r_angle)}, distance_r: {round(self.episode_r_distance)}")

            ego_x, ego_y = zip(*self.ego_positions)
            agent_x, agent_y = zip(*self.agent_positions)

            # Plot ego vehicle positions with smaller dots and no connecting line
            plt.scatter(ego_x, ego_y, s=0.25, c='b', marker='o', label='Ego Vehicle')
            # Plot agent vehicle positions with smaller dots and no connecting line
            plt.scatter(agent_x, agent_y, s=0.25, c='r', marker='o', label='Agent Vehicle')
            plt.xlabel('X-axis (meters)')
            plt.ylabel('Y-axis (meters)')

            # Add larger dots every 10 steps (1 second)
            for i in range(0, len(self.ego_positions), 10):
                plt.scatter(self.ego_positions[i][0], self.ego_positions[i][1], s=10, c='b', marker='o')
                plt.scatter(self.agent_positions[i][0], self.agent_positions[i][1], s=10, c='r', marker='o')

            # Calculate the necessary x and y limits to achieve a 3:2 aspect ratio
            min_x, max_x = plt.xlim()
            min_y, max_y = plt.ylim()
            width = max_x - min_x
            height = max_y - min_y
            desired_aspect_ratio = 3 / 2

            if width / height > desired_aspect_ratio:
                new_height = width / desired_aspect_ratio
                plt.ylim(min_y - (new_height - height) / 2, max_y + (new_height - height) / 2)
            else:
                new_width = height * desired_aspect_ratio
                plt.xlim(min_x - (new_width - width) / 2, max_x + (new_width - width) / 2)

            # Set the aspect ratio to be equal for both axes
            plt.gca().set_aspect('equal', adjustable='box')

            plt.legend()
            plt.savefig('positions_plot.png', dpi=300)  # Increase the resolution by setting the DPI
            plt.close()

        return np.array(observation), float(reward), done, dict()

    def reset(self, seed=None):
        self.state, observation = self._reset(seed)

        self.steps = 0

        return np.array(observation)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
    
    def _clamp_angle(self, theta):
        if theta > np.pi:
            theta -= 2 * np.pi
        elif theta < -np.pi:
            theta += 2 * np.pi

        return theta

    def _get_distance(self, vehicle1, vehicle2):
        return np.sqrt((vehicle1.x - vehicle2.x) ** 2 + (vehicle1.y - vehicle2.y) ** 2)

    def _vehicle_dynamics(self, s, a):
        max_v = 15.6464
        max_phi = 0.437
        min_a = -10.0
        max_a = 6.0
        max_steering_rate = 0.874

        acceleration = np.clip(a.acceleration, min_a, max_a)
        steering_rate = np.clip(a.steering_rate, -max_steering_rate, max_steering_rate)

        L = 2.6
        delta_t = 0.1

        x_prime = s.x + (s.v * delta_t + 0.5 * acceleration * delta_t * delta_t) * np.cos(s.theta)
        y_prime = s.y + (s.v * delta_t + 0.5 * acceleration * delta_t * delta_t) * np.sin(s.theta)
        theta_prime = s.theta + delta_t * s.v * np.tan(s.phi) / L
        v_prime = s.v + acceleration * delta_t 
        phi_prime = s.phi + steering_rate * delta_t
        t_prime = s.t + delta_t

        theta_prime = self._clamp_angle(theta_prime)
        v_prime = np.clip(v_prime, 0.0, max_v)
        phi_prime = np.clip(phi_prime, -max_phi, max_phi)

        return Vehicle(x_prime, y_prime, theta_prime, v_prime, phi_prime, t_prime)

    def _dynamics(self, s, a):
        ego = s[0]
        agent = s[1]

        agent_a = Action(acceleration=0.0, steering_rate=0.0)

        ego_prime = self._vehicle_dynamics(ego, a)
        agent_prime = self._vehicle_dynamics(agent, agent_a)

        return [ego_prime, agent_prime]

    def _reward(self, s, a, sp):
        ego = s[0]
        agent = s[1]
        ego_prime = sp[0]
        agent_prime = sp[1]

        delta_t = 0.1
        desired_distance = 15.0

        angle_to_agent = np.arctan2(agent_prime.y - ego.y, agent_prime.x - ego.x) - ego.theta
        angle_to_agent_prime = np.arctan2(agent_prime.y - ego_prime.y, agent_prime.x - ego_prime.x) - ego_prime.theta

        distance = self._get_distance(ego, agent_prime)
        distance_prime = self._get_distance(ego_prime, agent_prime)

        r_delta_distance = ((distance - distance_prime) / delta_t) / 15.6464
        r_delta_distance *= 1 if distance_prime > desired_distance else -2
        r_delta_distance *= 1 if r_delta_distance > 0 else 2
        r_delta_distance *= 50

        r_delta_angle_to_agent = ((angle_to_agent - angle_to_agent_prime) / delta_t) / 0.874
        r_delta_angle_to_agent *= 1 if angle_to_agent_prime > 0.0 else -1
        r_delta_angle_to_agent *= 1 if r_delta_angle_to_agent > 0 else 2
        r_delta_angle_to_agent *= 100

        r_distance = -abs(distance_prime - desired_distance)
        r_distance *= 1 if distance_prime > desired_distance else 10

        r_angle = -abs(angle_to_agent_prime)
        r_angle *= 50

        # print(f"r_angle: {round(r_angle)}, r_distance: {round(r_distance)}, r_delta_distance: {round(r_delta_distance)}, r_delta_angle_to_agent: {round(r_delta_angle_to_agent)}, r_a_smooth: {round(r_a_smooth)}, r_s_smooth: {round(r_s_smooth)} ")

        self.episode_r_distance += r_distance
        self.episode_r_angle += r_angle

        # return r_delta_distance + r_delta_angle_to_agent
        return r_distance + r_angle

    def _observation(self, a, sp):
        ego_prime = sp[0]
        agent_prime = sp[1]

        angle_to_agent = np.arctan2(agent_prime.y - ego_prime.y, agent_prime.x - ego_prime.x) - ego_prime.theta
        angle_to_agent = self._clamp_angle(angle_to_agent)

        agent_theta = agent_prime.theta - ego_prime.theta
        agent_theta = self._clamp_angle(agent_theta)

        distance = self._get_distance(ego_prime, agent_prime)

        return [ego_prime.v, ego_prime.phi, angle_to_agent, distance, agent_theta, agent_prime.v]

    def _reset(self, seed=None):
        np_random, seed = gym.utils.seeding.np_random(seed)

        ego_theta = np_random.uniform(-0.75, 0.75)
        ego_v = 7.0

        agent_x = np_random.uniform(10.0, 50.0)
        agent_theta = np_random.uniform(-0.75, 0.75)
        agent_v = 7.0
        agent_phi = np_random.uniform(-0.1, 0.1)

        ego = Vehicle(theta=ego_theta, v=ego_v)
        agent = Vehicle(x=agent_x, theta=agent_theta, v=agent_v, phi=agent_phi)

        # ego = Vehicle(v=7.0)
        # agent = Vehicle(x=30.0, v=7.0)
        s = [ego, agent]
        o = self._observation(Action(acceleration=0.0, steering_rate=0.0), s)

        self.episode_r_angle = 0
        self.episode_r_distance = 0

        self.ego_positions.clear()
        self.agent_positions.clear()

        return s, o

    def _map_action_index_to_obj(self, action_index):
        return Action(acceleration=action_index[0], steering_rate=action_index[1])

    def _generate(self, s, a):
        sp = self._dynamics(s, a)
        r = self._reward(s, a, sp)
        o = self._observation(a, sp)

        return sp, o, r