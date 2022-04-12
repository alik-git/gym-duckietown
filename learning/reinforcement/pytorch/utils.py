import random

import gym
import numpy as np
import torch


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self,max_size):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done))


    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1,1),
            "done": np.stack(dones).reshape(-1,1)
        }


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward


mbrl_config_dict = {
    'seed': 0,
    'device': 'cuda:0',
    'log_frequency_agent': 1000,
    'save_video': False,
    'debug_mode': False,
    'experiment': 'default',
    'root_dir': './exp',
    'algorithm': {
        'name': 'planet',
        'agent': {
            '_target_': 'mbrl.planning.TrajectoryOptimizerAgent',
            'action_lb': '???',
            'action_ub': '???',
            'planning_horizon': '${overrides.planning_horizon}',
            'optimizer_cfg': '${action_optimizer}',
            'replan_freq': 1,
            'keep_last_solution': False,
            'verbose': '${debug_mode}'},
        'num_initial_trajectories': 5,
        'action_noise_std': 0.3,
        'test_frequency': 25,
        'num_episodes': 1000,
        'dataset_size': 1000000},
    'dynamics_model': {
        '_target_': 'mbrl.models.PlaNetModel',
        'obs_shape': [3, 64, 64],
        'obs_encoding_size': 1024,
        'encoder_config': [[3, 32, 4, 2], [32, 64, 4, 2], [64, 128, 4, 2], [128, 256, 4, 2]],
        'decoder_config': [[1024, 1, 1], [[1024, 128, 5, 2], [128, 64, 5, 2], [64, 32, 6, 2], [32, 3, 6, 2]]],
        'action_size': '???',
        'hidden_size_fcs': 200,
        'belief_size': 200,
        'latent_state_size': 30,
        'device': '${device}',
        'min_std': 0.1,
        'free_nats': 3.0,
        'kl_scale': 1.0,
        'grad_clip_norm': 10.0},
    'overrides': {
        'env': 'duckietown_gym_env',
        'trial_length': 250,
        'action_noise_std': 0.3,
        'num_grad_updates': 100,
        'sequence_length': 50,
        'batch_size': 50,
        'free_nats': 3,
        'kl_scale': 1.0,
        'planning_horizon': 12,
        'cem_num_iters': 10,
        'cem_elite_ratio': 0.1,
        'cem_population_size': 1000,
        'cem_alpha': 0.0,
        'cem_clipped_normal': True},
    'action_optimizer': {
        '_target_': 'mbrl.planning.CEMOptimizer',
        'num_iterations': '${overrides.cem_num_iters}',
        'elite_ratio': '${overrides.cem_elite_ratio}',
        'population_size': '${overrides.cem_population_size}',
        'alpha': '${overrides.cem_alpha}',
        'lower_bound': '???',
        'upper_bound': '???',
        'return_mean_elites': True,
        'device': '${device}',
        'clipped_normal': '${overrides.cem_clipped_normal}'}}
