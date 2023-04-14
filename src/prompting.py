import numpy as np
import re

np.set_printoptions(suppress=True)

def distance_string(distance):
        return f"{distance:.1f} m"


def speed_string(velocity):
    return f"{velocity:.1f} m/s"


def angle_string(angle):
    degrees = abs(np.rad2deg(angle))
    direction = "" if degrees == 0 else f" to the {'left' if angle > 0 else 'right'}"

    return f"{degrees:.1f}°{direction}"


def get_short_prompt(observation):
    ego_velocity, steering, angle, distance, direction, agent_velocity = observation

    return f""""Our vehicle is going {speed_string(ego_velocity)} with a steering angle of {angle_string(steering)}. The other vehicle is {distance_string(distance)} away and is {angle_string(angle)}. It is going {speed_string(agent_velocity)} with a direction of {angle_string(direction)}." ->"""


def get_shortest_prompt(observation):
    ego_velocity, steering, angle, distance, direction, agent_velocity = observation

    return f"[{ego_velocity:.1f}, {np.rad2deg(steering):.1f}, {np.rad2deg(angle):.1f}, {distance:.1f}, {np.rad2deg(direction):.1f}, {agent_velocity:.1f}] ->"


def get_completion(action):
    acceleration, steering_rate = action

    steering_rate = np.rad2deg(steering_rate)

    return f" ({acceleration:.1f} m/s^2, {steering_rate:.1f}°/s)"


def extract_action(completion):
    result = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    acceleration = float(result[0])
    steering_rate = np.deg2rad(float(result[-1]))

    return acceleration, steering_rate