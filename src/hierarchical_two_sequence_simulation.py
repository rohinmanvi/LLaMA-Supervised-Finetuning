import numpy as np
from stable_baselines3 import PPO
from transformers import GenerationConfig

from driver_env import DriverEnv
from prompting import get_waypoint_sequence_shortest_prompt, extract_two_action
from model_handler import ModelHandler

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

pastelBlue = "#0072B2"
pastelRed = "#F5615C"
pastelGreen = "#009E73"
pastelPurple = "#8770FE"

def main():
    env = DriverEnv()
    model = PPO.load("models/Two_Waypoint_PPO/best_model")
    model_handler = ModelHandler("decapoda-research/llama-7b-hf")
    generation_config = GenerationConfig(max_new_tokens=40, do_sample=False)

    episodes = 1
    average_reward = run_episodes(env, model, model_handler, generation_config, episodes)
    print(f"Average reward: {average_reward}")
    print("Total angle error:", env.total_angle_error)
    print("Total distance error:", env.total_distance_error)

    ego_positions = env.ego_positions
    agent_positions = env.agent_positions

    env.close()

    env = DriverEnv()

    model = PPO.load("models/PPO/best_model")

    observation = env.reset(0)

    done = False

    while not done:
        action, _ = model.predict(observation)
        observation, reward, done, _ = env.step(action)

    expert_ego_positions = env.ego_positions

    env.close()

    ego_x, ego_y = zip(*ego_positions)
    agent_x, agent_y = zip(*agent_positions)
    expert_x, expert_y = zip(*expert_ego_positions)

    # Plot agent vehicle positions with smaller dots and no connecting line
    plt.scatter(expert_x, expert_y, s=0.25, c=pastelGreen, marker='o', label='Expert Vehicle')
    # Plot agent vehicle positions with smaller dots and no connecting line
    plt.scatter(agent_x, agent_y, s=0.25, c=pastelRed, marker='o', label='Agent Vehicle')
    # Plot ego vehicle positions with smaller dots and no connecting line
    plt.scatter(ego_x, ego_y, s=0.25, c=pastelBlue, marker='o', label='Ego Vehicle')
    plt.xlabel('X-axis (meters)')
    plt.ylabel('Y-axis (meters)')

    # Add larger dots every 10 steps (1 second)
    for i in range(0, len(ego_positions), 10):
        plt.scatter(expert_ego_positions[i][0], expert_ego_positions[i][1], s=10, c=pastelGreen, marker='o')
        plt.scatter(agent_positions[i][0], agent_positions[i][1], s=10, c=pastelRed, marker='o')
        plt.scatter(ego_positions[i][0], ego_positions[i][1], s=10, c=pastelBlue, marker='o')

    # Calculate the necessary x and y limits to achieve a 3:2 aspect ratio
    min_x, max_x = plt.xlim()
    min_y, max_y = plt.ylim()
    width = max_x - min_x
    height = max_y - min_y
    desired_aspect_ratio = 2 / 1

    if width / height > desired_aspect_ratio:
        new_height = width / desired_aspect_ratio
        plt.ylim(min_y - (new_height - height) / 2, max_y + (new_height - height) / 2)
    else:
        new_width = height * desired_aspect_ratio
        plt.xlim(min_x - (new_width - width) / 2, max_x + (new_width - width) / 2)

    # Set the aspect ratio to be equal for both axes
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig('positions_plot.png', dpi=300)  # Increase the resolution by setting the DPI
    plt.close()

def run_episodes(env, model, model_handler, generation_config, episodes):
    total_reward = 0

    for i in range(episodes):
        observation = env.reset(i)
        done = False
        print("\n\nReset")

        prompt_so_far = ""

        steps = 0
        waypoint_x = 0
        waypoint_y = 0
        new_waypoint_x = 0
        new_waypoint_y = 0

        while not done:
            waypoint_x, waypoint_y, new_waypoint_x, new_waypoint_y, prompt_so_far = update_waypoints(env, steps, waypoint_x, waypoint_y, new_waypoint_x, new_waypoint_y, model_handler, generation_config, observation, prompt_so_far)
            waypoint_observation = create_waypoint_observation(env, observation, waypoint_x, waypoint_y, new_waypoint_x, new_waypoint_y, steps)
            print(f"Observation: {np.round(waypoint_observation, 3)}")

            action, _ = model.predict(waypoint_observation)
            print(f"Action: {np.round(action, 3)}")

            observation, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

    return total_reward / episodes

def update_waypoints(env, steps, waypoint_x, waypoint_y, new_waypoint_x, new_waypoint_y, model_handler, generation_config, observation, prompt_so_far):
    if steps % 10 == 0:
        ego_state = env.state[0]
        x, y, theta = ego_state.x, ego_state.y, ego_state.theta
        x_diff, y_diff = waypoint_x - x, waypoint_y - y
        distance_to_previous_waypoint = np.sqrt(x_diff ** 2 + y_diff ** 2)
        print(f"Distance to previous waypoint: {distance_to_previous_waypoint}")

        prompt_so_far += get_waypoint_sequence_shortest_prompt(observation)

        response = model_handler.generate_text(
            peft_model='models/sequence-driver-shortest-2',
            text=prompt_so_far,
            generation_config=generation_config
        )

        response = response[len(prompt_so_far):response.rfind(")") + 1]

        print("============================================================================")
        print(prompt_so_far + "|" + response)
        print("============================================================================")

        prompt_so_far += response + "\n"

        distance, angle, new_distance, new_angle = extract_two_action(response)

        true_angle = angle + theta
        delta_x, delta_y = distance * np.cos(true_angle), distance * np.sin(true_angle)
        waypoint_x, waypoint_y = x + delta_x, y + delta_y

        new_true_angle = new_angle + theta
        new_delta_x, new_delta_y = new_distance * np.cos(new_true_angle), new_distance * np.sin(new_true_angle)
        new_waypoint_x, new_waypoint_y = x + new_delta_x, y + new_delta_y

    return waypoint_x, waypoint_y, new_waypoint_x, new_waypoint_y, prompt_so_far

def create_waypoint_observation(env, observation, waypoint_x, waypoint_y, new_waypoint_x, new_waypoint_y, steps):
    ego_state = env.state[0]
    x, y, theta = ego_state.x, ego_state.y, ego_state.theta
    x_diff, y_diff = waypoint_x - x, waypoint_y - y
    new_x_diff, new_y_diff = new_waypoint_x - x, new_waypoint_y - y

    angle = np.arctan2(y_diff, x_diff) - theta
    angle = env._clamp_angle(angle)
    distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

    new_angle = np.arctan2(new_y_diff, new_x_diff) - theta
    new_angle = env._clamp_angle(new_angle)
    new_distance = np.sqrt(new_x_diff ** 2 + new_y_diff ** 2)

    waypoint_time = (steps % 10) * 0.1

    return np.array([*observation[:2], angle, distance, new_angle, new_distance, waypoint_time])

if __name__ == "__main__":
    main()