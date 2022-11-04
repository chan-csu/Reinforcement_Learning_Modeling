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
import warnings
warnings.filterwarnings("ignore") 
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
                gamma=0.999,
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
    env.returns={agent.name:[] for agent in env.agents}
    for batch in range(env.batch_per_episode):
        env.batch_number=batch
        batch_obs,batch_obs_next,batch_acts, batch_log_probs, batch_rews=tk.rollout(env)
        
        for agent in env.agents:
            env.returns[agent.name].extend(batch_rews[agent.name])
            V, _ ,VP= agent.evaluate(batch_obs[agent.name],batch_obs_next[agent.name] ,batch_acts[agent.name])
            bootstrap_vals=batch_rews[agent.name]+agent.gamma*VP.detach()
            A_k = bootstrap_vals - V.detach()    
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)       
            for _ in range(agent.grad_updates):                                                       # ALG STEP 6 & 7
                V, curr_log_probs,VP = agent.evaluate(batch_obs[agent.name],batch_obs_next[agent.name],batch_acts[agent.name])
                bootstrap_vals=batch_rews[agent.name]+agent.gamma*VP
                ratios = torch.exp(curr_log_probs - batch_log_probs[agent.name])
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - agent.clip, 1 + agent.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, bootstrap_vals.detach())
                agent.optimizer_policy_.zero_grad()
                actor_loss.backward(retain_graph=True)
                agent.optimizer_policy_.step()
                agent.optimizer_value_.zero_grad()
                critic_loss.backward()
                agent.optimizer_value_.step()                                                            
    
    print(f"Episode {episode} finished")
    for agent in env.agents:
        print(f"{agent.name} return is:  {torch.FloatTensor(env.returns[agent.name]).sum()}")
