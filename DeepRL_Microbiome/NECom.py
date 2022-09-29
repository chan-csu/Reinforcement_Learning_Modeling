from mimetypes import init
import Toolkit as tk
import ToyModel as tm
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np


agent1=tk.Agent("agent1",
                model=tm.Toy_Model_NE_1,
                actor_network=tk.DDPGActor,
                critic_network=tk.DDPGCritic,
                optimizer_policy=torch.optim.Adam,
                optimizer_value=torch.optim.Adam,
                buffer=tk.Memory(max_size=100000),
                observables=['agent1','agent2','S',"A","B"],
                actions=['EX_A_sp1','EX_B_sp1'],
                gamma=0.999,
                update_batch_size=32,
                lr=0.0001
                )

agent2=tk.Agent("agent2",
                model=tm.Toy_Model_NE_2,
                actor_network=tk.DDPGActor,
                critic_network=tk.DDPGCritic,
                optimizer_policy=torch.optim.Adam,
                optimizer_value=torch.optim.Adam,
                observables=['agent1','agent2','S',"A","B"],
                actions=['EX_A_sp2','EX_B_sp2'],
                buffer=tk.Memory(max_size=100000),
                gamma=0.999,
                update_batch_size=32,
                lr=0.0001 
)

agents=[agent1,agent2]

env=tk.Environment(name="Toy-NECOM",
                    agents=agents,
                    extracellular_reactions=[],
                    initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
                    inlet_conditions={"S":10},
                    max_c={'S':100,
                           'agent1':10,  
                           'agent2':10,
                           'A':10,
                           'B':10,})
env.reset()
for epoch in range(1000):
    Batch=env.batch_step(env.generate_random_c(40))
    for i in Batch:
        for ind,ag in enumerate(env.agents):
            ag.buffer.push(i[0][ag.observables],i[2][ind],i[1][ind],i[3][ag.observables])
    for ind,ag in enumerate(env.agents):
        S,R,A,Sp=ag.buffer.sample(min(ag.update_batch_size,ag.buffer.buffer.__len__()))
        ag.optimizer_value_.zero_grad()
        Qvals = ag.critic_network_(torch.FloatTensor(S), torch.FloatTensor(A))
        next_actions = ag.actor_network_(torch.FloatTensor(Sp))
        next_Q = ag.target_critic_network_.forward(torch.FloatTensor(Sp), next_actions.detach())
        Qprime = torch.FloatTensor(np.expand_dims(np.array(R),1)) + next_Q
        critic_loss=nn.MSELoss()(Qvals,Qprime.detach())
        critic_loss.backward()
        ag.optimizer_value_.step()
        ag.optimizer_policy_.zero_grad()
        policy_loss = -ag.critic_network_(torch.FloatTensor(S), ag.actor_network_(torch.FloatTensor(S))).mean()
        policy_loss.backward()
        ag.optimizer_policy_.step()
        print("Epoch: ",epoch,"Agent: ",ag.name,"Critic Loss: ",critic_loss.item())
        for target_param, param in zip(ag.target_actor_network_.parameters(), ag.actor_network_.parameters()):
            target_param.data.copy_(param.data * ag.tau + target_param.data * (1.0 - ag.tau))
    
        for target_param, param in zip(ag.target_critic_network_.parameters(), ag.critic_network_.parameters()):
            target_param.data.copy_(param.data * ag.tau + target_param.data * (1.0 - ag.tau))