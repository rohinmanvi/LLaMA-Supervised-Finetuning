import numpy as np
import gym
from gym import spaces
from collections import namedtuple
import math
import json
import random
import re

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

        self.examples = []

        with open('data/new_waypoint_data.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                self.examples.append(data)

        self.max_steps = 20

        self.reset()

        max_v = 15.6464
        max_phi = 0.437
        max_steering_rate = 0.874
        min_a = -10.0
        max_a = 6.0
        max_d = 50
        max_angle = np.pi
        max_time = 0.1 * self.max_steps

        self.action_space = spaces.Box(
            low = np.array([min_a, -max_steering_rate]), 
            high = np.array([max_a, max_steering_rate]), 
            shape = (2,),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low = np.array([0.0, -max_phi, -max_angle, -max_d, -max_angle, -max_d, 0.0]), 
            high = np.array([max_v, max_phi, max_angle, max_d, max_angle, max_d, max_time]), 
            shape = (7,),
            dtype = np.float32
        )

    def step(self, action):
        action = self._map_action_index_to_obj(action)
        self.state, observation, reward = self._generate(self.state, action)

        self.steps += 1
        done = self.steps >= self.max_steps

        return np.array(observation), float(reward), done, dict()

    def reset(self):
        self.state, observation = self._reset()

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
        new_agent = s[2]

        ego_prime = self._vehicle_dynamics(ego, a)

        return [ego_prime, agent, new_agent]

    def _reward(self, s, a, sp):
        ego = s[0]
        ego_prime = sp[0]
        agent_prime = sp[1]
        new_agent_prime = sp[2]

        if ego_prime.t > 1.0:
            agent_prime = new_agent_prime

        delta_t = 0.1

        distance = self._get_distance(ego, agent_prime)
        distance_prime = self._get_distance(ego_prime, agent_prime)

        r_delta_distance = ((distance - distance_prime) / delta_t) / 15.6464
        r_delta_distance *= 1 if r_delta_distance > 0 else 2

        r_a_smooth = -abs(a.acceleration) / 6
        r_a_smooth *= 0.2

        r_s_smooth = -abs(a.steering_rate) / 0.874
        r_s_smooth *= 0.1

        r_distance = 0.0
        if round(ego_prime.t, 2) == 1.0 or round(ego_prime.t, 2) == 2.0:
            print("waypoint reached")
            r_distance = -distance_prime

        print((r_delta_distance, r_a_smooth, r_s_smooth, r_distance))

        return r_delta_distance + r_a_smooth + r_s_smooth

    def _observation(self, a, sp):
        ego_prime = sp[0]
        agent_prime = sp[1]
        new_agent_prime = sp[2]

        angle_to_agent = np.arctan2(agent_prime.y - ego_prime.y, agent_prime.x - ego_prime.x) - ego_prime.theta
        angle_to_agent = self._clamp_angle(angle_to_agent)

        distance = self._get_distance(ego_prime, agent_prime)

        new_angle_to_agent = np.arctan2(new_agent_prime.y - ego_prime.y, new_agent_prime.x - ego_prime.x) - ego_prime.theta
        new_angle_to_agent = self._clamp_angle(new_angle_to_agent)

        new_distance = self._get_distance(ego_prime, new_agent_prime)

        return [ego_prime.v, ego_prime.phi, angle_to_agent, distance, new_angle_to_agent, new_distance, ego_prime.t]

    def _reset(self):
        example = random.choice(self.examples)
        prompt = example['prompt']
        completion = example['completion']

        reg = r"[-+]?\d*\.\d+|\d+"

        result = re.findall(reg, completion)

        distance = float(result[0])
        angle = np.deg2rad(float(result[1]))

        new_distance = float(result[2])
        new_angle = np.deg2rad(float(result[3]))

        agent_x = distance * np.cos(angle)
        agent_y = distance * np.sin(angle)

        new_agent_x = new_distance * np.cos(new_angle)
        new_agent_y = new_distance * np.sin(new_angle)

        result = re.findall(reg, prompt)

        ego_v = float(result[0])
        ego_phi = np.deg2rad(float(result[1]))

        ego = Vehicle(v=ego_v, phi=ego_phi)
        agent = Vehicle(x=agent_x, y=agent_y)
        new_agent = Vehicle(x=new_agent_x, y=new_agent_y)

        s = [ego, agent, new_agent]
        o = self._observation(Action(acceleration=0.0, steering_rate=0.0), s)

        return s, o

    def _map_action_index_to_obj(self, action_index):
        return Action(acceleration=action_index[0], steering_rate=action_index[1])

    def _generate(self, s, a):
        sp = self._dynamics(s, a)
        r = self._reward(s, a, sp)
        o = self._observation(a, sp)

        return sp, o, r