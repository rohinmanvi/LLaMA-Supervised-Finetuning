import numpy as np
from stable_baselines3 import PPO
from transformers import GenerationConfig

from driver_env import DriverEnv
from prompting import get_waypoint_sequence_shorter_prompt, extract_two_action
from model_handler import ModelHandler

np.set_printoptions(suppress=True)

def main():
    env = DriverEnv()
    model = PPO.load("models/Two_Waypoint_PPO/best_model")
    model_handler = ModelHandler("decapoda-research/llama-7b-hf")
    generation_config = GenerationConfig(max_new_tokens=40, do_sample=False)

    episodes = 1
    average_reward = run_episodes(env, model, model_handler, generation_config, episodes)
    print(f"Average reward: {average_reward}")

def run_episodes(env, model, model_handler, generation_config, episodes):
    total_reward = 0

    for i in range(episodes):
        observation = env.reset()
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

        prompt_so_far += get_waypoint_sequence_shorter_prompt(observation)

        response = model_handler.generate_text(
            peft_model='models/sequence-driver-shorter',
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