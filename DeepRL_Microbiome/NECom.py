from mimetypes import init
import Toolkit as tk
import ToyModel as tm
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import time
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
#                 lr_critic=0.0001,
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
#                 lr_critic=0.0001
# )

# agents=[agent1,agent2]

# env=tk.Environment(name="Toy-NECOM",
#                     agents=agents,
#                     extracellular_reactions=[],
#                     initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
#                     inlet_conditions={"S":10},
#                     max_c={'S':100,
#                            'agent1':10,  
#                            'agent2':10,
#                            'A':10,
#                            'B':10,},
#                            dt=0.05)


# env.reset()
# for epoch in range(1000):
#     Batch=env.batch_step(env.generate_random_c(8))
#     for i in Batch:
#         for ind,ag in enumerate(env.agents):
#             ag.buffer.push(i[0][ag.observables],i[2][ind],i[1][ind],i[3][ag.observables])
#     for ind,ag in enumerate(env.agents):
#         S,R,A,Sp=ag.buffer.sample(min(ag.buffer_sample_size,ag.buffer.buffer.__len__()))
#         ag.optimizer_value_.zero_grad()
#         Qvals = ag.critic_network_(torch.FloatTensor(S), torch.FloatTensor(A))
#         next_actions = ag.actor_network_(torch.FloatTensor(Sp))
#         next_Q = ag.target_critic_network_.forward(torch.FloatTensor(Sp), next_actions.detach())
#         Qprime = torch.FloatTensor(np.expand_dims(np.array(R),1)) + next_Q
#         print(np.sum(R))
#         critic_loss=nn.MSELoss()(Qvals,Qprime.detach())
#         critic_loss.backward()
#         ag.optimizer_value_.step()
#         ag.optimizer_policy_.zero_grad()
#         policy_loss = -ag.critic_network_(torch.FloatTensor(S), ag.actor_network_(torch.FloatTensor(S))).mean()
#         policy_loss.backward()
#         ag.optimizer_policy_.step()
#         print("Epoch: ",epoch,"Agent: ",ag.name,"Critic Loss: ",critic_loss.item())
#         for target_param, param in zip(ag.target_actor_network_.parameters(), ag.actor_network_.parameters()):
#             target_param.data.copy_(param.data * ag.tau + target_param.data * (1.0 - ag.tau))
    
#         for target_param, param in zip(ag.target_critic_network_.parameters(), ag.critic_network_.parameters()):
#             target_param.data.copy_(param.data * ag.tau + target_param.data * (1.0 - ag.tau))
# for episode in range(1000):
#     env.reset()

#     for agent in env.agents:
#         agent.epsilon=1
#         agent.rewards=[]
#     for ep in range(1000):
#         s,r,a,sp=env.step()
#         for ind,ag in enumerate(env.agents):
#             ag.buffer.push(s[ag.observables],r[ind],a[ind],sp[ag.observables])
#             ag.rewards.append(r[ind])
#             s_,r_,a_,sp_=ag.buffer.sample(min(ag.buffer_sample_size,ag.buffer.buffer.__len__()))
#             ag.optimizer_value_.zero_grad()
#             Qvals = ag.critic_network_(torch.FloatTensor(s_), torch.FloatTensor(a_))
#             next_actions = ag.actor_network_(torch.FloatTensor(sp_))
#             next_Q = ag.target_critic_network_.forward(torch.FloatTensor(sp_), next_actions.detach())
#             Qprime = torch.FloatTensor(np.expand_dims(np.array(r_),1)) + next_Q
#             critic_loss=nn.MSELoss()(Qvals,Qprime.detach())
#             critic_loss.backward()
#             ag.optimizer_value_.step()
#             ag.optimizer_policy_.zero_grad()
#             policy_loss = -ag.critic_network_(torch.FloatTensor(s_), ag.actor_network_(torch.FloatTensor(s_))).mean()
#             policy_loss.backward()
#             ag.optimizer_policy_.step()
#             for target_param, param in zip(ag.target_actor_network_.parameters(), ag.actor_network_.parameters()):
#                 target_param.data.copy_(param.data * ag.tau + target_param.data * (1.0 - ag.tau))
#             for target_param, param in zip(ag.target_critic_network_.parameters(),ag.critic_network_.parameters()):
#                 target_param.data.copy_(param.data * ag.tau + target_param.data * (1.0 - ag.tau))
            
#     for agent in env.agents:
#         print(episode)
#         print(np.sum(agent.rewards))
#         print(f'{agent.name} : {agent.actor_network_(torch.FloatTensor(env.state[agent.observables])).detach().numpy()}')
    
# for episode in range(1000):
#     env.reset()

#     for agent in env.agents:
#         agent.rewards=[]
#     C=[]
#     episode_len=1000
#     for ep in range(episode_len):
#         env.t=episode_len-ep
#         s,r,a,sp=env.step()
        
#         for ind,ag in enumerate(env.agents):
#             ag.rewards.append(r[ind])
#             ag.optimizer_value_.zero_grad()
#             Qvals = ag.critic_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), torch.FloatTensor(a[ind]))
#             next_actions = ag.actor_network_(torch.FloatTensor(np.hstack([sp[ag.observables],env.t-1])))
#             if env.t==1:
#                 next_Q = torch.FloatTensor([0])
#             else:
#                 next_Q = ag.critic_network_.forward(torch.FloatTensor(np.hstack([sp[ag.observables],env.t-1])), next_actions.detach())

            
            
#             Qprime = torch.FloatTensor(np.expand_dims(np.array(r[ind]),0))+ag.gamma*next_Q
#             critic_loss=nn.MSELoss()(Qvals,Qprime.detach())
#             critic_loss.backward()
#             ag.optimizer_value_.step()
#             ag.optimizer_policy_.zero_grad()
#             policy_loss = -ag.critic_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), ag.actor_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])))).mean()
#             policy_loss.backward()
#             ag.optimizer_policy_.step()
#         C.append(env.state.copy()) 

#     pd.DataFrame(C,columns=env.species).to_csv("Data.csv")

#     for agent in env.agents:
#         print(episode)
#         print(np.sum(agent.rewards))    
    
    








# for episode in range(10000):
#     env.reset()

#     for agent in env.agents:
#         agent.epsilon=0.01+0.2*np.exp(-episode/50)
#         agent.rewards=[]
#     C=[]
#     for ep in range(1000):
#         s,r,a,sp=env.step()
#         for ind,ag in enumerate(env.agents):
#             ag.rewards.append(r[ind])
#             ag.optimizer_value_.zero_grad()
#             Qvals = ag.critic_network_(torch.FloatTensor(s[ag.observables]), torch.FloatTensor(a[ind]))
#             next_actions = ag.actor_network_(torch.FloatTensor(sp[ag.observables]))
#             next_Q = ag.critic_network_.forward(torch.FloatTensor(sp[ag.observables]), next_actions.detach())
#             Qprime = torch.FloatTensor(np.expand_dims(np.array(r[ind]),0)) + next_Q
#             td_error=r[ind]-ag.R-Qvals.detach()+Qprime.detach()
#             ag.R=ag.R+ag.alpha*td_error.detach()
#             critic_loss=nn.MSELoss()(Qvals,r[ind]-ag.R+Qprime.detach())
#             critic_loss.backward()
#             ag.optimizer_value_.step()
#             ag.optimizer_policy_.zero_grad()
#             policy_loss = -ag.critic_network_(torch.FloatTensor(s[ag.observables]), ag.actor_network_(torch.FloatTensor(s[ag.observables]))).mean()
#             policy_loss.backward()
#             ag.optimizer_policy_.step()
#         C.append(env.state.copy()) 
#     pd.DataFrame(C,columns=env.species).to_csv("Data.csv")
#     for agent in env.agents:
#         print(episode)
#         print(np.sum(agent.rewards))


agent1=tk.Agent("agent1",
                model=tm.ToyModel_SA.copy(),
                actor_network=tk.DDPGActor,
                critic_network=tk.DDPGCritic,
                reward_network=tk.Reward,
                optimizer_policy=torch.optim.Adam,
                optimizer_value=torch.optim.Adam,
                optimizer_reward=torch.optim.Adam,
                buffer=tk.Memory(max_size=100000),
                observables=['agent1', 'Glc', 'Starch'],
                actions=["Amylase_Ex"],
                gamma=0.99,
                update_batch_size=8,
                lr_actor=0.00001,
                lr_critic=0.0001,
                tau=0.1
                )

agents=[agent1]

env=tk.Environment(name="Toy-Exoenzyme",
                    agents=agents,
                    dilution_rate=0.01,
                    
                    initial_condition={"Glc":100,"agent1":0.1,"Starch":10},
                    inlet_conditions={"Starch":10},
                    extracellular_reactions=[{"reaction":{
                    "Glc":10,
                    "Starch":-0.1,},
                    "kinetics": (lambda x,y: 0.01*x*y/(10+x),("Glc","Amylase"))}]
                    ,
                    max_c={'Glc':100,
                           'agent1':10,  
                           'Starch':10,
                           },
                           dt=0.1)








for episode in range(1000):
    env.reset()

    for agent in env.agents:
        agent.rewards=[]
    C=[]
    episode_len=1000
    for ep in range(episode_len):
        env.t=episode_len-ep
        s,r,a,sp=env.step()
        
        
        for ind,ag in enumerate(env.agents):
            ag.rewards.append(r[ind])
            # ag.optimizer_reward_.zero_grad()
            # # r_pred=ag.reward_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), torch.FloatTensor(a[ind]))
            # # r_loss=nn.MSELoss()(r_pred,torch.FloatTensor(np.expand_dims(np.array(r[ind]),0)))
            # # r_loss.backward()
            # ag.optimizer_reward_.step()
            ag.optimizer_value_.zero_grad()
            Qvals = ag.critic_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), torch.FloatTensor(a[ind]))
            next_actions = ag.actor_network_(torch.FloatTensor(np.hstack([sp[ag.observables],env.t-1])))
            if env.t==1:
                next_Q = torch.FloatTensor([0])
            else:
                next_Q = ag.critic_network_.forward(torch.FloatTensor(np.hstack([sp[ag.observables],env.t-1])), next_actions.detach())
            Qprime = torch.FloatTensor(np.expand_dims(np.array(r[ind]),0))+ag.gamma*next_Q
            critic_loss=nn.MSELoss()(Qvals,Qprime.detach())
            critic_loss.backward()
            ag.optimizer_value_.step()
            ag.optimizer_policy_.zero_grad()
            policy_loss = -ag.critic_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), ag.actor_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])))).mean()
            policy_loss.backward()
            ag.optimizer_policy_.step()
        C.append(env.state.copy()) 

    pd.DataFrame(C,columns=env.species).to_csv("Data.csv")

    for agent in env.agents:
        print(episode)
        print(np.sum(agent.rewards))    
    
    




