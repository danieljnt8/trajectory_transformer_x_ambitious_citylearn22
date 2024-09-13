import numpy as np
import time
from datasets import Dataset
from agents.orderenforcingwrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv

class Constants:
    episodes = 1
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

def action_space_to_dict(aspace):
    """ Only for box space """
    return {
        "high": aspace.high,
        "low": aspace.low,
        "shape": aspace.shape,
        "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {
        "action_space": action_space_dicts,
        "observation_space": observation_space_dicts,
        "building_info": building_info,
        "observation": observations
    }
    return obs_dict

def run_episode(env, agent, num_steps, episode_metrics, action_list, reward_list):
    obs_dict = env_reset(env)
    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    interrupted = False

    try:
        while episodes_completed < Constants.episodes:
            action_list.append(actions)
            observations, rewards, done, _ = env.step(actions)
            reward_list.append(rewards)

            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {
                    "price_cost": metrics_t[0],
                    "emission_cost": metrics_t[1],
                    "grid_cost": metrics_t[2]
                }

                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contact organizers")
                
                episode_metrics.append(metrics)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}")

                obs_dict = env_reset(env)
                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter() - step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(observations)
                agent_time_elapsed += time.perf_counter() - step_start

            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True

    return episodes_completed, num_steps, action_list, reward_list, episode_metrics

def process_observations(agent, num_agents=5, num_timesteps=8760):
    final_averaged_observations = np.zeros((num_agents, num_timesteps, 33))
    calculators = agent.central_agent.calculators

    for agent_ in range(num_agents):
        for t in range(num_timesteps):
            total_observations = 0
            for calculator in calculators:
                total_observations += np.array(calculator.observations_data[agent_][t])
            final_averaged_observations[agent_][t] = total_observations / len(calculators)

    return final_averaged_observations[:, :-1, :-1].reshape(-1, 32)

def save_dataset(final_observation, action_list, reward_list, num_agents=5):
    actions_arr = np.array(action_list)
    rewards_arr = np.array(reward_list)
    array_size = 8759
    dones_array = np.zeros(array_size)
    dones_array[-1] = 1
    dones_array = np.tile(dones_array, num_agents)

    actions_arr_res = actions_arr.reshape(-1, num_agents).T.reshape(-1, 1, 1)
    rewards_arr_res = rewards_arr.reshape(-1, num_agents).T.reshape(-1, )

    data_dict = {
        'observations': final_observation,
        'actions': actions_arr_res,
        'rewards': rewards_arr_res,
        'dones': dones_array,
    }

    dataset = Dataset.from_dict(data_dict)
    dataset.save_to_disk("winner_dataset_phase_1.pkl")
    print("Dataset saved to 'winner_dataset_phase_1.pkl'")

def main():
    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    num_steps = 0
    action_list = []
    reward_list = []
    episode_metrics = []

    episodes_completed, num_steps, action_list, reward_list, episode_metrics = run_episode(
        env, agent, num_steps, episode_metrics, action_list, reward_list
    )

    final_observation = process_observations(agent)
    save_dataset(final_observation, action_list, reward_list)

if __name__ == "__main__":
    main()