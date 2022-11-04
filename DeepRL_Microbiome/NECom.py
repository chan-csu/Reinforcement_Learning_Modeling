from mimetypes import init
from turtle import color
import Toolkit as tk
import ToyModel as tm
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import ray
import seaborn  as sns
import matplotlib.pyplot as plt
# agent1=tk.Agent("agent1",
#                 model=tm.Toy_Model_NE_1,
#                 actor_network=tk.DDPGActor,
#                 critic_network=tk.DDPGCritic,
#                 reward_network=tk.Reward,
#                 optimizer_policy=torch.optim.Adam,
#                 optimizer_value=torch.optim.Adam,
#                 optimizer_reward=torch.optim.Adam,
#                 buffer=tk.Memory(max_size=100000),
#                 observables=['agent1','agent2','S',"A","B"],
#                 actions=['EX_A_sp1','EX_B_sp1'],
#                 gamma=0.99,
#                 update_batch_size=8,
#                 lr_actor=0.000001,
#                 lr_critic=0.001,
#                 tau=0.1
#                 )

# agent2=tk.Agent("agent2",
#                 model=tm.Toy_Model_NE_2,
#                 actor_network=tk.DDPGActor,
#                 critic_network=tk.DDPGCritic,
#                 reward_network=tk.Reward,
#                 optimizer_policy=torch.optim.Adam,
#                 optimizer_value=torch.optim.Adam,
#                 optimizer_reward=torch.optim.Adam,
#                 observables=['agent1','agent2','S',"A","B"],
#                 actions=['EX_A_sp2','EX_B_sp2'],
#                 buffer=tk.Memory(max_size=100000),
#                 gamma=0.99,
#                 update_batch_size=8,
#                 tau=0.1,
#                 lr_actor=0.000001,
#                 lr_critic=0.001
# )

# agents=[agent1,agent2]

# env=tk.Environment(name="Toy-NECOM",
#                     agents=agents,
#                     extracellular_reactions=[],
#                     initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
#                     inlet_conditions={"S":100},
#                     max_c={'S':100,
#                            'agent1':10,  
#                            'agent2':10,
#                            'A':10,
#                            'B':10,},
#                            dt=0.05,dilution_rate=0.01)




agent1=tk.Agent("agent1",
                model=tm.ToyModel_SA.copy(),
                actor_network=tk.NN,
                critic_network=tk.NN,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
                observables=['agent1', 'Glc', 'Starch'],
                actions=["Amylase_Ex"],
                gamma=1,
                update_batch_size=8,
                lr_actor=0.0000001,
                lr_critic=0.0001,
                tau=0.1
                )

agents=[agent1]

env=tk.Environment(name="Toy-Exoenzyme",
                    agents=agents,
                    dilution_rate=0.0001,
                    initial_condition={"Glc":100,"agent1":0.1,"Starch":10},
                    inlet_conditions={"Starch":10},
                    extracellular_reactions=[{"reaction":{
                    "Glc":10,
                    "Starch":-0.1,},
                    "kinetics": (lambda x,y: x*y/(10+x),("Glc","Amylase"))}]
                    ,
                    max_c={'Glc':100,
                           'agent1':10,  
                           'Starch':10,
                           },
                           dt=0.1,
                           )


for episode in range(env.number_of_episodes):
    env.reset()
    for batch in range(env.batch_per_episode):
        env.batch_number=batch
        batch_obs, batch_acts, batch_log_probs, batch_rtgs=tk.rollout(env)
        for agent in env.agents:
            V, _ = env.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()    
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)       
            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
		    	# Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
		    	# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
		    	# NOTE: we just subtract the logs, which is the same as
		    	# dividing the values and then canceling the log with e^log.
		    	# For why we use log probabilities instead of actual probabilities,
		    	# here's a great explanation: 
		    	# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
		    	# TL;DR makes gradient ascent easier behind the scenes.
		    	ratios = torch.exp(curr_log_probs - batch_log_probs)
		    	# Calculate surrogate losses.
		    	surr1 = ratios * A_k
		    	surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
		    	# Calculate actor and critic losses.
		    	# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
		    	# the performance function, but Adam minimizes the loss. So minimizing the negative
		    	# performance function maximizes it.
		    	actor_loss = (-torch.min(surr1, surr2)).mean()
		    	critic_loss = nn.MSELoss()(V, batch_rtgs)
		    	# Calculate gradients and perform backward propagation for actor network
		    	self.actor_optim.zero_grad()
		    	actor_loss.backward(retain_graph=True)
		    	self.actor_optim.step()
		    	# Calculate gradients and perform backward propagation for critic network
		    	self.critic_optim.zero_grad()
		    	critic_loss.backward()
		    	self.critic_optim.step()                                                            
    
        
 
