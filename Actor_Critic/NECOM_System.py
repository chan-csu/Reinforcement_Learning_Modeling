
# Script for running Community Dynamic Flux Balance Analysis (CDFBA)
# Written by: Parsa Ghadermazi
from cmath import inf
from dataclasses import dataclass,field
import datetime
from tkinter import HIDDEN
from xmlrpc.client import DateTime
import numpy as np
import cobra
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import pickle
import pandas
#import cplex
from ToyModel import  Toy_Model_NE_1,Toy_Model_NE_2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple,deque
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import warnings
import torch.autograd
from torch.autograd import Variable
import gym
warnings.filterwarnings("ignore")
Scaler=StandardScaler()
HIDDEN_SIZE=20
NUMBER_OF_BATCHES=1000
Main_dir = os.path.dirname(os.path.abspath(__file__))

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

        
class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state):
        experience = (state, action, np.array([reward]), next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            
        
        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)



class DDPGActor(nn.Module):

    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, act_size),
            nn.Tanh() )

    def forward(self, x):
       return self.net(x)

class DDPGCritic(nn.Module):       
    
    def __init__(self, obs_size, act_size):

        super(DDPGCritic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
             nn.Linear(40, 40),
            nn.ReLU(), nn.Linear(40, 40),
            nn.ReLU(), nn.Linear(40, 40)
            
            )


        self.out_net = nn.Sequential(
                       nn.Linear(40 + act_size, 100),
                       nn.ReLU(),
                       nn.Linear(100, 100),
                       nn.ReLU(),
                       nn.Linear(100, 1)
                       )
    
    def forward(self, x, a):
        obs = self.obs_net(x)           
        return self.out_net(torch.cat([obs, a],dim=1))

def main(Models: list = [Toy_Model_NE_1.copy(), Toy_Model_NE_2.copy()], max_time: int = 100, Dil_Rate: float = 0.000000001, alpha: float = 0.01, Starting_Q: str = "FBA"):
    """
    This is the main function for running dFBA.
    The main requrement for working properly is
    that the models use the same notation for the
    same reactions.

    Starting_Policy:

    Defult --> Random: Initial Policy will be a random policy for all agents.
    Otherwise --> a list of policies, pickle file addresses, for each agent.


    """
    # Adding Agents info ###-----------------------------------------------------

    # State dimensions in this RLDFBA variant include: [Agent1,...,Agentn, glucose,starch]
    for i in range(len(Models)):
        if not hasattr(Models[i], "_name"):
            Models[i].NAME = "Agent_" + str(i)
            print(f"Agent {i} has been given a defult name")
        Models[i].solver.objective.name = "_pfba_objective"
    # -------------------------------------------------------------------------------

    # Mapping internal reactions to external reactions, and operational parameter
    # setup ###-------------------------------------------------------------------

    # For more information about the structure of the ODEs,see ODE_System function
    # or the documentation.

    Mapping_Dict = Build_Mapping_Matrix(Models)
    Init_C = np.ones((len(Models)+len(Mapping_Dict["Ex_sp"]),))
    Inlet_C = np.zeros((len(Models)+len(Mapping_Dict["Ex_sp"]),))

    #Parameters that are use inside DFBA

    Params = {
        "Dilution_Rate": Dil_Rate,
        "Inlet_C": Inlet_C,
        "Agents_Index": [i for i in range(len(Models))],
    }

    #Define Agent attributes
    Obs=[i for i in range(len(Models))]
    Obs.extend([Mapping_Dict["Ex_sp"].index(sp)+len(Models) for sp in Mapping_Dict["Ex_sp"] if sp!='P' ])
    for ind,m in enumerate(Models):
        m.observables=Obs
        m.actions=(Mapping_Dict["Mapping_Matrix"][Mapping_Dict["Ex_sp"].index("A"),ind],Mapping_Dict["Mapping_Matrix"][Mapping_Dict["Ex_sp"].index("B"),ind])
        m.policy=DDPGActor(len(m.observables),len(m.actions))
        m.policy_target=DDPGActor(len(m.observables),len(m.actions))
        m.value=DDPGCritic(len(m.observables),len(m.actions))
        m.value_target=DDPGCritic(len(m.observables),len(m.actions))
        m.R=0
        m.tau=0.01
        m.optimizer_policy=optim.Adam(params=m.policy.parameters(), lr=0.0001)
        m.optimizer_policy_target=optim.Adam(params=m.policy.parameters(), lr=0.01)
        m.optimizer_value=optim.Adam(params=m.value.parameters(), lr=0.001)
        m.optimizer_value_target=optim.Adam(params=m.value.parameters(), lr=0.01)
        m.Net_Obj=nn.MSELoss()
        m.epsilon=0.1
        m.buffer=Memory(10000)
        m.alpha=0.01
        m.update_batch=50
        
    ### I Assume that the environment states are all observable. Env states will be stochastic
    Params["Env_States"]=Models[0].observables
    Params["Env_States_Initial_Ranges"]=[[0.1,0.1+0.00000001],[0.1,0.1+0.00000001],[100,100+0.00001],[0.0000000001,0.00000001+0.00000000001],[0.00000001,0.00000001+0.00000000001]]
    for i in range(len(Models)):
        Init_C[i] = 0.001
        #Models[i].solver = "cplex"
    # writer = SummaryWriter(comment="-DeepRLDFBA_NECOM")
    Outer_Counter = 0


    

    Generate_Batch(dFBA, Params, Init_C, Models, Mapping_Dict)
        # Batch_Out=list(map(list, zip(*Batch_Out)))
    #     for index,Model in enumerate(Models):
    #         Model.optimizer.zero_grad()
    #         action_scores_v = Model.Policy(obs_v)
    #         loss_v = Model.Net_Obj(action_scores_v, acts_v)
    #         loss_v.backward()
    #         Model.optimizer.step()
    #         print(f"{Model.NAME}")
    #         print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (c, loss_v.item(), reward_m, reward_b))

    #         writer.add_scalar(f"{Model.NAME} reward_mean", reward_m, c)
    
    Time=datetime.datetime.now().strftime("%d_%m_%Y.%H_%M_%S")
    Results_Dir=os.path.join(Main_dir,"Outputs",str(Time))
    os.mkdir(Results_Dir)
    with open(os.path.join(Results_Dir,"Models.pkl"),'wb') as f:
        pickle.dump(Models,f)


def dFBA(Models, Mapping_Dict, Init_C, Params, t_span, dt=0.1):
    """
    This function calculates the concentration of each species
    Models is a list of COBRA Model objects
    Mapping_Dict is a dictionary of dictionaries
    """
    ##############################################################
    # Initializing the ODE Solver
    ##############################################################
    t = np.arange(t_span[0], t_span[1], dt)
    ##############################################################
    # Solving the ODE
    ##############################################################
    for m in Models:
        m.episode_reward=0
        m.episode_steps=[]

    
    sol, t = odeFwdEuler(ODE_System, Init_C, dt,  Params,
                         t_span, Models, Mapping_Dict)
    
    for m in Models:
        m.Episode=Episode(reward=m.episode_reward, steps=m.episode_steps)




    return [m.Episode for m in Models]


def ODE_System(C, t, Models, Mapping_Dict, Params, dt,Counter):
    """
    This function calculates the differential equations for the system
    Models is a list of COBRA Model objects
    NOTE: this implementation of DFBA is compatible with RL framework
    Given a policy it will genrate episodes. Policies can be either deterministic or stochastic
    Differential Equations Are Formatted as follows:
    [0]-Models[1]
    [1]-Models[2]
    []-...
    [n-1]-Models[n]
    [n]-Exc[1]
    [n+1]-Exc[2]
    []-...
    [n+m-1]-Exc[m]
    [n+m]-Starch
    """
    C[C < 0] = 0
    dCdt = np.zeros(C.shape)
    Sols = list([0 for i in range(Models.__len__())])
    for i,M in enumerate(Models):

        if random.random()<M.epsilon:
            
            # M.a=M.policy(torch.FloatTensor([C[M.observables]])).detach().numpy()[0]
            # M.rand_act=np.random.uniform(low=-(M.a+1), high=1-M.a,size=len(M.actions)).copy()
            # M.a+=M.rand_act
            M.a=np.random.uniform(low=-1, high=1,size=len(M.actions))
        else:
            M.a=M.policy(torch.FloatTensor([C[M.observables]])).detach().numpy()[0]
        
        for index,item in enumerate(Mapping_Dict["Ex_sp"]):
            if Mapping_Dict['Mapping_Matrix'][index,i]!=-1:
                M.reactions[Mapping_Dict['Mapping_Matrix'][index,i]].upper_bound=20
                M.reactions[Mapping_Dict['Mapping_Matrix'][index,i]].lower_bound=-General_Uptake_Kinetics(C[index+len(Models)])
                
            
        for index,flux in enumerate(M.actions):

            if M.a[index]<0:
            
                M.reactions[M.actions[index]].lower_bound=M.a[index]*abs(M.reactions[M.actions[index]].lower_bound)

            else:

                M.reactions[M.actions[index]].lower_bound=M.a[index]*M.reactions[M.actions[index]].upper_bound

            

        Sols[i] = Models[i].optimize()

        if Sols[i].status == 'infeasible':
            Models[i].reward= -10
            dCdt[i] = 0

        else:
            dCdt[i] += Sols[i].objective_value*C[i]
            Models[i].reward =Sols[i].objective_value



    ### Writing the balance equations

    for i in range(Mapping_Dict["Mapping_Matrix"].shape[0]):
        for j in range(len(Models)):
            if Mapping_Dict["Mapping_Matrix"][i, j] != -1:
                if Sols[j].status == 'infeasible':
                    dCdt[i] = 0
                else:
                    dCdt[i+len(Models)] += Sols[j].fluxes.iloc[Mapping_Dict["Mapping_Matrix"]
                                                                    [i, j]]*C[j]
    dCdt += np.array(Params["Dilution_Rate"])*(Params["Inlet_C"]-C)
    Next_C=C+dCdt*dt
    for m in Models:
        m.buffer.push(torch.FloatTensor([C[m.observables]]).detach().numpy()[0],m.a,m.reward,torch.FloatTensor([Next_C[m.observables]]).detach().numpy()[0])
        if Counter>0 and Counter%m.update_batch==0:
            # TD_Error=[]
            S,A,R,Sp=m.buffer.sample(50)
            Qvals = m.value(torch.FloatTensor(S), torch.FloatTensor(A))
            next_actions = m.policy(torch.FloatTensor(Sp))
            next_Q = m.value_target.forward(torch.FloatTensor(Sp), next_actions.detach())
            Qprime = torch.FloatTensor(R) + next_Q
            critic_loss=m.Net_Obj(Qvals,Qprime)
            policy_loss = -m.value(torch.FloatTensor(S), m.policy(torch.FloatTensor(S))).mean()
            m.optimizer_policy.zero_grad()
            policy_loss.backward()
            m.optimizer_policy.step()

            m.optimizer_value.zero_grad()
            critic_loss.backward() 
            m.optimizer_value.step()
            

            for target_param, param in zip(m.policy_target.parameters(), m.policy.parameters()):
                target_param.data.copy_(param.data * m.tau + target_param.data * (1.0 - m.tau))
       
            for target_param, param in zip(m.value_target.parameters(), m.value.parameters()):
                target_param.data.copy_(param.data * m.tau + target_param.data * (1.0 - m.tau))
        
        
        
        
        
        
        
        
        
        
        
        m.episode_reward+=m.reward

    
    
    return dCdt


def Build_Mapping_Matrix(Models):
    """
    Given a list of COBRA model objects, this function will build a mapping matrix

    """

    Ex_sp = []
    Temp_Map={}
    for model in Models:
        
        
        if not hasattr(model,"Biomass_Ind"):
            raise Exception("Models must have 'Biomass_Ind' attribute in order for the DFBA to work properly!")
        
        
        for Ex_rxn in model.exchanges :
            if Ex_rxn!=model.reactions[model.Biomass_Ind]:
                if list(Ex_rxn.metabolites.keys())[0].id not in Ex_sp:
                    Ex_sp.append(list(Ex_rxn.metabolites.keys())[0].id)
                if list(Ex_rxn.metabolites.keys())[0].id in Temp_Map.keys():
                   Temp_Map[list(Ex_rxn.metabolites.keys())[0].id][model]=Ex_rxn
                else:
                     Temp_Map[list(Ex_rxn.metabolites.keys())[0].id]={model:Ex_rxn}

    Mapping_Matrix = np.zeros((len(Ex_sp), len(Models)), dtype=int)
    for i, id in enumerate(Ex_sp):
        for j, model in enumerate(Models):
            if model in Temp_Map[id].keys():
                Mapping_Matrix[i, j] = model.reactions.index(Temp_Map[id][model].id)
            else:
                Mapping_Matrix[i, j] = -1
    return {"Ex_sp": Ex_sp, "Mapping_Matrix": Mapping_Matrix}






def Starch_Degradation_Kinetics(a_Amylase: float, Starch: float, Model="", k: float = 1):
    """
    This function calculates the rate of degradation of starch
    a_Amylase Unit: mmol
    Starch Unit: mg

    """

    return a_Amylase*Starch*k/(Starch+10)


def Glucose_Uptake_Kinetics(Glucose: float, Model=""):
    """
    This function calculates the rate of glucose uptake
    ###It is just a simple imaginary model: Replace it with better model if necessary###
    Glucose Unit: mmol

    """
    return 20*(Glucose/(Glucose+20))


def General_Uptake_Kinetics(Compound: float, Model=""):
    """
    This function calculates the rate of uptake of a compound in the reactor
    ###It is just a simple imaginary model: Replace it with better model if necessary###
    Compound Unit: mmol

    """
    return 100*(Compound/(Compound+20))





def odeFwdEuler(ODE_Function, ICs, dt, Params, t_span, Models, Mapping_Dict):
    Integrator_Counter = 0
    t = np.arange(t_span[0], t_span[1], dt)
    sol = np.zeros((len(t), len(ICs)))
    sol[0] = ICs
    for i in range(1, len(t)):
        sol[i] = sol[i-1] + \
            ODE_Function(sol[i-1], t[i-1], Models, Mapping_Dict,
                         Params, dt,Integrator_Counter)*dt
        Integrator_Counter += 1
    return sol, t


def Generate_Batch(dFBA, Params, Init_C, Models, Mapping_Dict,t_span=[0, 100], dt=0.1):


    Init_C[list(Params["Env_States"])] = [random.uniform(Range[0], Range[1]) for Range in Params["Env_States_Initial_Ranges"]]

    
    for BATCH in range(NUMBER_OF_BATCHES):
        for model in Models:
            model.epsilon=0.01+0.99/(np.exp(BATCH/20))
        dFBA(Models, Mapping_Dict, Init_C, Params, t_span, dt=dt)
    
        for mod in Models:
            print(f"{BATCH} - {mod.NAME} earned {mod.episode_reward} during this episode!")
    





def Flux_Clipper(Min,Number,Max):
    return(min(max(Min,Number),Max))
    

if __name__ == "__main__":
    # Init_Pols=[]
    # for i in range(2):
    #     Init_Pols.append(os.path.join(Main_dir,"Outputs","Agent_"+str(i)+"_3900.pkl"))

    # cProfile.run("","Profile")
    main([Toy_Model_NE_1.copy(),Toy_Model_NE_2.copy()])

