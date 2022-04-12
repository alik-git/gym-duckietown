import ast
import argparse
import logging

import os
from tkinter import W
import numpy as np

from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Duckietown Specific
from learning.reinforcement.pytorch.ddpg import DDPG
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper

import wandb
import torch
    
import errno
import os
from datetime import datetime

import hydra
import omegaconf
from learning.reinforcement.pytorch.utils import planet_config_dict

cfg = omegaconf.OmegaConf.create(planet_config_dict)

# ADD THE MBRL LIB TO PATH SO YOU CAN IMPORT IT 
import sys
sys.path.append('/home/kuwajerw/repos/duckietown-mbrl-lib')


import mbrl.algorithms.planet as planet
import mbrl.util.env
import mbrl.constants
from mbrl.env.termination_fns import no_termination
from mbrl.models import ModelEnv, ModelTrainer
from mbrl.planning import RandomAgent, create_trajectory_optim_agent_for_model

# folder in which to save logs
# will make timestamped folder in here
base_logdir = 'saved_runs'

# timestamp stuff
base_day_str = datetime.now().strftime('%Y-%m-%d')
base_time_str = datetime.now().strftime('%H-%M-%S')
curr_timestamp_str = base_day_str + '_' + base_time_str

# made this helper function to manually save the logs 
# in addition to the W&B logs just in case
def make_save_folder(label=''):

    # make timestamp logdir path
    base_path = os.getcwd() + f'/{base_logdir}/'
    mydir = os.path.join(
        base_path, 
        base_day_str, base_time_str + "_" + label)

    # actually create the folder
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    
    return mydir + "/"


def save_list(items, label, dest):
    with open(os.path.join(dest, label), 'w') as f:
        f.writelines(items)



# curr_timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

video_name = f"{curr_timestamp_str}.mp4"



def _enjoy():          

    # save run logs in folder that matches the wandb run name
    logdir = make_save_folder(label=wandb.run.name)
    
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = Monitor(env, logdir, force=True)
    print("Initialized Wrappers")
    
    # record video
    video_rec = VideoRecorder(env, logdir + video_name)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # PLANET STUFF ##############################
        
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    # Create PlaNet model
    rng = torch.Generator(device=cfg.device)
    rng.manual_seed(cfg.seed)
    cfg.dynamics_model.action_size = env.action_space.shape[0]

    planet = hydra.utils.instantiate(cfg.dynamics_model)
    assert isinstance(planet, mbrl.models.PlaNetModel)
    planet.load('learning/reinforcement/pytorch/models/planet.pth')
    model_env = ModelEnv(env, planet, no_termination, generator=rng)
    trainer = ModelTrainer(planet, optim_lr=1e-3, optim_eps=1e-4)
    
    agent = create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent)
    # PLANET STUFF ##############################

    # Initialize policy
    # policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    # policy.load(filename='ddpg', directory='learning/reinforcement/pytorch/models/')

    obs = env.reset()
    done = False
    reward_list = []
    action_list = []

    # PLANET STUFF ##############################
    agent.reset()
    planet.reset_posterior()
    action = None
    # PLANET STUFF ##############################

    # while True:
    # while not done:
    for step in range(100):
        
        # PLANET STUFF ##############################
        planet.update_posterior(obs, action=action, rng=rng)
        action_noise = 0
        action = agent.act(obs) + action_noise
        action = np.clip(action, -1.0, 1.0)  # to account for the noise
        # PLANET STUFF ##############################

        # action = policy.predict(np.array(obs))
        # action = env.action_space.sample()
        # Perform action
        obs, reward, done, _ = env.step(action)

        # log stuff with wandb
        wandb.log({'step' : step})
        wandb.log({'action0': action[0], 'action1' : action[1], 'global_step': step})
        wandb.log({'reward' : reward, 'global_step': step})
        wandb.log({'done' : int(done), 'global_step': step})

        # env.render() # optional
        video_rec.capture_frame() # record video

        # maintain manual log just in case 
        reward_list.append(str(reward) + '\n')
        action_list.append(str(action) + '\n')
        
    
    # cleanup here
    # done = False
    # obs = env.reset()
    video_rec.close()
    env.close()

    # save manual log 
    save_list(reward_list, f"rewards_{curr_timestamp_str}.txt", logdir)
    save_list(action_list, f"actions_{curr_timestamp_str}.txt", logdir)
    
        

if __name__ == '__main__':
    wandb.init(project="MBRL_Duckyt", entity="mbrl_ducky", monitor_gym=True)
    _enjoy()
    
