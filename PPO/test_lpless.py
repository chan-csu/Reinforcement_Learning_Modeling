from mimetypes import init
from turtle import color
import flux_explorer as tk
import ToyModel as tm
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import time
import ray
import os
import seaborn  as sns
import matplotlib.pyplot as plt
import warnings
import json
import multiprocessing as mp
NUM_CORES = 8
print(f"{NUM_CORES} cores available: Each policy evaluation will\ncontain {NUM_CORES} Episode(s)")
warnings.filterwarnings("ignore") 

agent1_rew_vect=torch.zeros(len(tm.ToyModel_SA.reactions),)
agent1_rew_vect[tm.ToyModel_SA.Biomass_Ind]=1
model1=tm.ToyModel_SA.copy()
# model2=tm.ToyModel_SA.copy()





agent1=tk.Agent("agent1",
				model=model1,
				actor_network=tk.NN,
				critic_network=tk.NN,
                reward_vect=agent1_rew_vect,
				biomass_ind=tm.ToyModel_SA.Biomass_Ind,
				clip=0.05,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=1,
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1', 'Glc', 'Starch'],
				gamma=0.1,
				)

# agent2=tk.Agent("agent2",
# 				model=tm.Toy_Model_NE_2,
# 				actor_network=tk.NN,
# 				critic_network=tk.NN,
# 				biomass_ind=tm.Toy_Model_NE_2.Biomass_Ind,
#                 reward_vect=agent2_rew_vect,
# 				clip=0.1,
# 				lr_actor=0.0001,
# 				lr_critic=0.001,
# 				grad_updates=1,
# 				optimizer_actor=torch.optim.Adam,
# 				optimizer_critic=torch.optim.Adam,       
# 				observables=['agent1','agent2','S',"A","B"],
# 				gamma=1
# )

agents=[agent1]

env=tk.Environment(name="Toy-NECOM-lpless_single_0.1",
					agents=agents,
					dilution_rate=0.0001,
					initial_condition={"Glc":100,"agent1":0.1},
					inlet_conditions={"Glc":100},
                    extracellular_reactions=[{"reaction":{
                     "Glc":10,
                     "Starch":-0.1,},
                     "kinetics": (tk.general_kinetic,("Glc","Amylase"))}],
							dt=0.1,
							episode_time=20,
							number_of_batches=10000,
							episodes_per_batch=NUM_CORES,)

# with open(f"Results/Toy-NECOM-host/agent1_0.pkl", 'rb') as f:
#        agent1 = pickle.load(f)

# with open(f"Results/Toy-NECOM-host/agent2_0.pkl", 'rb') as f:
#        agent2 = pickle.load(f)


env.agents=[agent1]

env.rewards={agent.name:[] for agent in env.agents}

if not os.path.exists(f"Results/{env.name}"):
	os.makedirs(f"Results/{env.name}")

for agent in env.agents:
    agent.model.solver="glpk"


for batch in range(env.number_of_batches):
	batch_obs,batch_acts, batch_log_probs, batch_rtgs=tk.rollout(env)
	for agent in env.agents:
		V, _,mean= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
		A_k = batch_rtgs[agent.name] - V.detach()   
		A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5) 
		for _ in range(agent.grad_updates):                                                      
			V, curr_log_probs,mean = agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
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
	
	if batch%500==0:
		
		with open(f"Results/{env.name}/{env.name}_{batch}.pkl", 'wb') as f:
			pickle.dump(env, f)
		with open(f"Results/{env.name}/observations_{batch}.pkl", 'wb') as f:
			pickle.dump(batch_obs, f)
		
		with open(f"Results/{env.name}/actions_{batch}.pkl", 'wb') as f:
			pickle.dump(batch_acts, f)
	
		
		# with open(f"Results/{env.name}/returns_{batch}.json", 'w') as f:
		# 	json.dump(env.rewards, f)


		


	print(f"Batch {batch} finished:")
	for agent in env.agents:
		print(f"{agent.name} return is:  {np.mean(env.rewards[agent.name][-env.episodes_per_batch:])}")

