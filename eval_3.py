import numpy as np
import time
from tqdm.auto import trange
from omegaconf import OmegaConf
from tqdm.auto import trange
from trajectory.planning.beam import beam_plan, batch_beam_plan
from trajectory.models.gpt import GPT
from trajectory.utils.common import set_seed
from trajectory.utils.env import create_env, rollout, vec_rollout
import torch
import os 
"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.orderenforcingwrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv

class Constants:
    episodes = 1
    schema_path = './data/citylearn_challenge_2022_phase_3/schema.json'

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
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
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict

run_config = "configs/medium/city_learn_traj_winner.yaml"
run_config = OmegaConf.load(run_config)

config = "configs/eval_base.yaml"
config = OmegaConf.load(config)



def run(ts_path= ["checkpoints/city_learn/winner/quantile/",
        "phase_1/run"]):
    checkpoints_path = "".join(ts_path)

    beam_context_size = 24 #config.beam_context 5 , before 20
    beam_width = 32 #config.beam_width 32, before 16
    beam_steps = 3 #config.beam_steps 5 , before = 3
    plan_every = config.plan_every
    sample_expand = config.sample_expand
    k_act = config.k_act
    k_obs = config.k_obs
    k_reward = config.k_reward
    temperature = config.temperature
    discount = 0.99  #config.discount
    max_steps = 8759 
    device = "cuda" 

    num_envs= 5


    value_placeholder=1e6
    value_placeholder = np.ones((num_envs, 1)) * value_placeholder


    discretizer = torch.load(os.path.join(checkpoints_path, "discretizer.pt"), map_location=device)

    model = GPT(**run_config.model)
    

    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoints_path, "model_last.pt"), map_location=device))

    transition_dim, obs_dim, act_dim = model.transition_dim, model.observation_dim, model.action_dim
    context = torch.zeros(5, model.transition_dim * (max_steps + 1), dtype=torch.long).to(device)


    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    obs = agent.register_reset(obs_dict,return_actions = False)
    state = np.array(obs)
    state_input = state[:,:-1]
    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []

    obs_tokens = discretizer.encode(state_input, subslice=(0, obs_dim))
    context[:, :obs_dim] = torch.as_tensor(obs_tokens, device=device)  # initial tokens for planning

    total_rewards = np.zeros(num_envs)
    dones = np.zeros(num_envs, dtype=np.bool)

    list_actions = []
    for step in trange(max_steps, desc="Rollout steps", leave=True):
        if step % plan_every == 0:
            context_offset = transition_dim * (step + 1) - act_dim - 2
            context_not_dones = context[~dones, :context_offset]
            prediction_tokens = batch_beam_plan(
                    model, discretizer, context_not_dones, steps=beam_steps, beam_width=beam_width, 
                    context_size=beam_context_size,k_act=k_act, k_obs=k_obs, k_reward=k_reward, 
                temperature=temperature, discount=discount, sample_expand=sample_expand
                )
            plan_buffer = torch.zeros(num_envs, prediction_tokens.shape[-1], dtype=torch.long, device=device)
            plan_buffer[~dones] = prediction_tokens
        else:
            plan_buffer = plan_buffer[:, transition_dim:]
        action_tokens = plan_buffer[:, :act_dim]
        action = discretizer.decode(action_tokens.cpu().numpy(), subslice=(obs_dim, obs_dim + act_dim))
        obs, reward, done, _ = env.step(action)
        list_actions.append(action)
        obs_trans = agent.compute_action(obs, return_actions = False)
        state = np.array(obs_trans)
        state_input = state[:,:-1]

        obs_tokens = discretizer.encode(state_input[~dones], subslice=(0, obs_dim))
        reward = np.array(reward)
        reward_tokens = discretizer.encode(
                np.hstack([reward.reshape(-1, 1), value_placeholder]),
                subslice=(transition_dim - 2, transition_dim)
            )

        context_offset = model.transition_dim * step
        context[~dones, context_offset + obs_dim:context_offset + obs_dim + act_dim] = torch.as_tensor(action_tokens[~dones], device=device)
        context[~dones, context_offset + transition_dim - 2:context_offset + transition_dim] = torch.as_tensor(reward_tokens[~dones], device=device)
        context[~dones, context_offset + model.transition_dim:context_offset + model.transition_dim + model.observation_dim] = torch.as_tensor(obs_tokens, device=device)
        total_rewards[~dones] += reward[~dones]

        dones[done] = True
        if np.all(dones):
            break
    list_act_array = np.array(list_actions)
    np.save(checkpoints_path+"/list_act_array_3.npy",list_act_array)

if __name__ == "__main__":
    
    for i in range(1,6):
        if i == 1:
            ts_path =  ["checkpoints/city_learn/winner/quantile/", "phase_1_1000_epochs/run"]
        else:
            ts_path =  ["checkpoints/city_learn/winner/quantile/", f"phase_1_1000_epochs/run_{i}"]
        
        run(ts_path)