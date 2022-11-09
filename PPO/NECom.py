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
import plotext as plt
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
                clip=0.01,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=5,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
                observables=['agent1', 'Glc', 'Starch'],
                actions=["Amylase_Ex"],
                gamma=1,
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
                    "kinetics": (lambda x,y: x*y/(10+x),("Glc","Amylase"))}],
                    max_c={'Glc':100,
                           'agent1':10,  
                           'Starch':10,
                           },
                           dt=0.1,
                           episode_time=500,
                           number_of_batches=5000,
                           episodes_per_batch=5,
                           )


# for episode in range(env.number_of_episodes):
#     env.reset()
#     env.returns={agent.name:[] for agent in env.agents}
#     # for agent in env.agents:
#     #     # agent.cov_var = torch.full(size=(len(agent.actions),), fill_value=2*np.exp(-episode/50))
#     #     # agent.cov_mat = torch.diag(agent.cov_var)
#     for batch in range(env.batch_per_episode):
#         env.batch_number=batch
#         batch_obs,batch_obs_next,batch_acts, batch_log_probs, batch_rews=tk.rollout(env)
        
#         for agent in env.agents:
#             env.returns[agent.name].extend(batch_rews[agent.name])
#             V, _ ,VP= agent.evaluate(batch_obs[agent.name],batch_obs_next[agent.name] ,batch_acts[agent.name])
#             bootstrap_vals=batch_rews[agent.name]+agent.gamma*VP.detach()
#             A_k = bootstrap_vals - V.detach()   
#             A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5) 
#             V, curr_log_probs,_ = agent.evaluate(batch_obs[agent.name],batch_obs_next[agent.name],batch_acts[agent.name])
#             for _ in range(agent.grad_updates):                                                      
#                 V, curr_log_probs,_ = agent.evaluate(batch_obs[agent.name],batch_obs_next[agent.name],batch_acts[agent.name])
#                 bootstrap_vals=batch_rews[agent.name]+agent.gamma*VP.detach()
#                 ratios = torch.exp(curr_log_probs - batch_log_probs[agent.name])
#                 surr1 = ratios * A_k.detach()
#                 surr2 = torch.clamp(ratios, 1 - agent.clip, 1 + agent.clip) * A_k
#                 actor_loss = (-torch.min(surr1, surr2)).mean()
#                 critic_loss = nn.MSELoss()(V, bootstrap_vals.detach())
#                 agent.optimizer_policy_.zero_grad()
#                 actor_loss.backward(retain_graph=True)
#                 agent.optimizer_policy_.step()
#                 agent.optimizer_value_.zero_grad()
#                 critic_loss.backward()
#                 agent.optimizer_value_.step()                                                            
    
#     print(f"Episode {episode} finished")
#     for agent in env.agents:
#         print(f"{agent.name} return is:  {torch.FloatTensor(env.returns[agent.name]).sum()}")

env.rewards={agent.name:[] for agent in env.agents}
for batch in range(env.number_of_batches):
    batch_obs,batch_acts, batch_log_probs, batch_rtgs=tk.rollout(env)
    for agent in env.agents:
        V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
        A_k = batch_rtgs[agent.name] - V.detach()   
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5) 
        for _ in range(agent.grad_updates):                                                      
            V, curr_log_probs = agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
            ratios = torch.exp(curr_log_probs - batch_log_probs[agent.name])
            surr1 = ratios * A_k.detach()
            surr2 = torch.clamp(ratios, 1 - agent.clip, 1 + agent.clip) * A_k
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs[agent.name])
            agent.optimizer_policy_.zero_grad()
            actor_loss.backward(retain_graph=False)
            agent.optimizer_policy_.step()
            agent.optimizer_value_.zero_grad()
            critic_loss.backward()
            agent.optimizer_value_.step()                                                            
    
    print(f"Batch {batch} finished:")
    for index_ag,agent in enumerate(env.agents):
        plt.clt() # to clear the terminal
        plt.cld()
        plt.clf()
        plt.scatter(env.rewards[agent.name])
    plt.show()
