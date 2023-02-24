import cobra
import Toolkit as tk
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import time
import ray
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import json
import multiprocessing as mp
import rich

NUM_CORES = mp.cpu_count()

warnings.filterwarnings("ignore")
model_base = cobra.io.read_sbml_model("iML1515.xml")
medium = model_base.medium.copy()
agent1_ko=("trpC",)
agent2_ko=("leuB","thrC","hisB","pheA")
knockouts_gene_names=agent1_ko+agent2_ko
unique_kos = ((agent1_ko[0],agent2_ko[i]) for i in range(1,len(agent2_ko)))
gene_ids = {}
for ko_gene in knockouts_gene_names:
    for gene in model_base.genes:
        if gene.name == ko_gene:
            gene_ids[ko_gene] = gene.id
            break

exchane_reactions = {
    "serA": "EX_ser__L_e",
    "glyA": "EX_gly_e",
    "thrC": "EX_thr__L_e",
    "ilvA": "EX_ile__L_e",
    "trpC": "EX_trp__L_e",
    "cysE": "EX_cys__L_e",
    "pheA": "EX_phe__L_e",
    "tyrA": "EX_tyr__L_e",
    "hisB": "EX_his__L_e",
    "proA": "EX_pro__L_e",
    "argA": "EX_arg__L_e",
    "leuB": "EX_leu__L_e",
}

exchange_species = {}
exchange_mets = {}
for i in exchane_reactions.items():
    exchange_mets[i[0]] = list(model_base.reactions.get_by_id(i[1]).metabolites.keys())[
        0
    ].id

ic={
    key.lstrip("EX_"):10000 for key,val in model_base.medium.items() 
}
ic['glc__D_e']=200
ic['agent1']=0.5
ic['agent2']=0.5

for ko in unique_kos:
    model1 = model_base.copy()
    model2 = model_base.copy()
    model1.remove_reactions(model1.genes.get_by_id(gene_ids[ko[0]]).reactions)
    model2.remove_reactions(model2.genes.get_by_id(gene_ids[ko[1]]).reactions)
    model1.Biomass_Ind=model1.reactions.index("BIOMASS_Ec_iML1515_core_75p37M")
    model2.Biomass_Ind=model2.reactions.index("BIOMASS_Ec_iML1515_core_75p37M")
    model1.solver = "gurobi"
    model2.solver = "gurobi"
    if model1.optimize().objective_value != 0 or model2.optimize().objective_value != 0:
        rich.print(f"[yellow]Skipping {ko} because at least one organism can grow without auxotrophy")
        continue
    else:
        rich.print(f"[green]Non of the KOs can grow without auxotrophy: Running {ko}")
    ko_name = ko[0] + "_" + ko[1]
    agent1 = tk.Agent(
        "agent1",
        model=model1,
        actor_network=tk.NN,
        critic_network=tk.NN,
        clip=0.1,
        lr_actor=0.0001,
        lr_critic=0.001,
        grad_updates=1,
        optimizer_actor=torch.optim.Adam,
        optimizer_critic=torch.optim.Adam,
        observables=[
            "agent1",
            "agent2",
            "glc__D_e",
            exchange_mets[ko[0]],
            exchange_mets[ko[1]],
        ],
        actions=[exchane_reactions[ko[0]], exchane_reactions[ko[1]]],
        gamma=1,
    )
    agent2 = tk.Agent(
        "agent2",
        model=model2,
        actor_network=tk.NN,
        critic_network=tk.NN,
        clip=0.1,
        lr_actor=0.0001,
        lr_critic=0.001,
        grad_updates=1,
        optimizer_actor=torch.optim.Adam,
        optimizer_critic=torch.optim.Adam,
        observables=[
            "agent1",
            "agent2",
            "glc__D_e",
            exchange_mets[ko[0]],
            exchange_mets[ko[1]],
        ],
        actions=[exchane_reactions[ko[0]], exchane_reactions[ko[1]]],
        gamma=1,
    )

    env = tk.Environment(
        ko_name+"_pub",
        agents=[agent1, agent2],
        dilution_rate=0,
        extracellular_reactions=[],
        initial_condition=ic,
        inlet_conditions={},
        max_c={},
        dt=0.1,
        episode_time=20,  ##TOBECHANGED
        number_of_batches=2000,  ##TOBECHANGED
        episodes_per_batch=NUM_CORES,
    )

    env.rewards = {agent.name: [] for agent in env.agents}

    if not os.path.exists(f"Results/aa_ecoli/{env.name}"):
        os.makedirs(f"Results/aa_ecoli/{env.name}")

    for batch in range(env.number_of_batches):
        rich.print(f"[green]Started batch: {batch}")
        batch_obs, batch_acts, batch_log_probs, batch_rtgs = tk.rollout(env)
        for agent in env.agents:
            V, _ = agent.evaluate(batch_obs[agent.name], batch_acts[agent.name])
            A_k = batch_rtgs[agent.name] - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5)
            for _ in range(agent.grad_updates):
                V, curr_log_probs = agent.evaluate(
                    batch_obs[agent.name], batch_acts[agent.name]
                    )
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
        if batch % 1 == 0:
            # for agent in env.agents:
            #     # with open(f"Results/aa_ecoli/{env.name}/{agent.name}_{batch}.pkl", "wb") as f:
            #     #     pickle.dump(agent, f)
            pd.DataFrame(env.rewards).to_csv(f"Results/aa_ecoli/{env.name}/rewards.csv")

            with open(f"Results/aa_ecoli/{env.name}/batch_obs_{batch}.pkl", "wb") as f:
                pickle.dump(batch_obs, f)
            with open(f"Results/aa_ecoli/{env.name}/batch_acts_{batch}.pkl", "wb") as f:
                pickle.dump(batch_acts, f)
        if batch == env.number_of_batches - 1:
            with open(f"Results/aa_ecoli/{env.name}/final_env.pkl", "wb") as f:
                pickle.dump(env, f)


        
        
        print(f"Batch {batch} finished:")
        for agent in env.agents:
            print(
            f"{agent.name} return is:  {np.mean(env.rewards[agent.name][-env.episodes_per_batch:])}"
            )
            