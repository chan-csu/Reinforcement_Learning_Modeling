import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import cobra

class Net(nn.Module):
    
    
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)



class DFBA_Env(gym.Env):
    """A simulation environment for dynamic flux balance analysis.
    
    Assumption 1: The environment can expose all of the extracellular
                  species, microorganisms and metabolites, concentrations. The agents might 
                  not be able to observe them all though.
    
    Note: I'm deliberately deviating from the gym's common practice.
    I will move the actions to agents
    """
    def __init__(self, Agents:cobra.Model,dt:float):
        super(DFBA_Env, self).__init__()
        self.Agents = Agents
        self.dt=dt

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = net(obs_v)
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs