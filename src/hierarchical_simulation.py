import numpy as np
import math

from stable_baselines3 import PPO

from driver_env import DriverEnv

np.set_printoptions(suppress=True)

env = DriverEnv()

model = PPO.load("models/PPO_Waypoint/best_model")

# average_reward = 0
episodes = 1

for i in range(episodes):

    observation = env.reset()
    done = False

    print("Reset")
    # print(f"Observation: {np.round(observation, 3)}")

    time = 0.0

    prev_acceleration = 0.0
    prev_steering_rate = 0.0

    while not done:

        ego_state = env.state[0]
        x = ego_state.x
        y = ego_state.y
        theta = ego_state.theta

        print((x, y, theta))

        if time % 1.0 == 0.0:
            delta_x = 6.0
            delta_y = 1.0

            waypoint_x = x + delta_x
            waypoint_y = y + delta_y

            print((x, y))
            print((waypoint_x, waypoint_y))

        velocity, steering, _, _, _, _ = observation

        x_diff = waypoint_x - x
        y_diff = waypoint_y - y

        angle = np.arctan2(y_diff, x_diff) - theta
        angle = env._clamp_angle(angle)

        distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

        waypoint_time = time % 1.0

        waypoint_observation = np.array([velocity, steering, prev_acceleration, prev_steering_rate, angle, distance, waypoint_time])

        print(f"waypoint_observation: {np.round(waypoint_observation, 3)}")

        action, _ = model.predict(waypoint_observation)
        observation, reward, done, _ = env.step(action)
        # average_reward += reward

        prev_acceleration, prev_steering_rate = action

        time += 0.1

        # print(f"Observation: {np.round(observation, 3)}")

# average_reward /= episodes
# print(f"average reward: {average_reward}")