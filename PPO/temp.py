
from mimetypes import init
from turtle import color
import Toolkit as tk
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
import rich
import multiprocessing as mp
import cobra

NUM_CORES = 8
warnings.filterwarnings("ignore") 

agent1=tk.Agent("agent1",
				model=tm.Toy_Model_NE_Aux_1,
				actor_network=tk.NN,
				critic_network=tk.NN,
				clip=0.1,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=5,
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1','agent2','S',"A","B"],
				actions=['A_e','B_e'],
				gamma=1,
				)
agent2=tk.Agent("agent2",
				model=tm.Toy_Model_NE_Aux_2,
				actor_network=tk.NN,
				critic_network=tk.NN,
				clip=0.1,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=5,
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1','agent2','S',"A","B"],
				actions=['A_e','B_e'],
				gamma=1)
agents=[agent1,agent2]

env=tk.Environment(name="Toy-NECOM-Auxotrophs",
 					agents=agents,
 					dilution_rate=0.0001,
 					extracellular_reactions=[],
 					initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
 					inlet_conditions={"S":100},
 					max_c={'S':100,
 						   'agent1':10,  
 						   'agent2':10,
 						   'A':10,
 						   'B':10,},
 							dt=0.1,
 							episode_time=100,
 							number_of_batches=5000,
 							episodes_per_batch=int(NUM_CORES/2),)

env.rewards={agent.name:[] for agent in env.agents}
if not os.path.exists(f"Results/{env.name}"):
	os.makedirs(f"Results/{env.name}")
for agent in env.agents:
	agent.model.solver="glpk"
for batch in range(env.number_of_batches):
	batch_obs,batch_acts, batch_log_probs, batch_rtgs=tk.rollout(env)
	for agent in env.agents:
		V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
		A_k = batch_rtgs[agent.name] - V.detach()   
		A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5) 
		if batch==0:
			rich.print("[bold yellow] Hold on, bringing the creitc network to range...[/bold yellow]")
			err=51
			while err>50:
				V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
				critic_loss = nn.MSELoss()(V, batch_rtgs[agent.name])
				agent.optimizer_value_.zero_grad()
				critic_loss.backward()
				agent.optimizer_value_.step() 
				err=critic_loss.item()
				rich.print(f"[bold yellow] Error: {err}[/bold yellow]")
			rich.print("[bold green] Done![/bold green]")
		else: 			
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
	if batch%500==0:		
		with open(f"Results/{env.name}/{env.name}_{batch}.pkl", 'wb') as f:
			pickle.dump(env, f)
		with open(f"Results/{env.name}/observations_{batch}.pkl", 'wb') as f:
			pickle.dump(batch_obs,f)		
		with open(f"Results/{env.name}/actions_{batch}.pkl", 'wb') as f:
			pickle.dump(batch_acts,f)		
    		
	print(f"Batch {batch} finished:")
	for agent in env.agents:
		print(f"{agent.name} return is:  {np.mean(env.rewards[agent.name][-env.episodes_per_batch:])}")		