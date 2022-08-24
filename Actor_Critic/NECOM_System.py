
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
warnings.filterwarnings("ignore")
Scaler=StandardScaler()
HIDDEN_SIZE=20
NUMBER_OF_BATCHES=100
Main_dir = os.path.dirname(os.path.abspath(__file__))
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


@dataclass
class Buffer:
    """
    A dataclass for feeding raw observations efficiently to pytorch.

    Observations should be given in the form of (State,Actions,Reward)
    """
    window:int=10
    batch:list[tuple]=field(default_factory=lambda:[(i,i,i) for i in range(10)])

    def __post_init__(self):
        self.queue=deque(self.batch,self.window)
    
    def update_queue(self,interaction:tuple):
        self.queue.appendleft(interaction)
    


class DDPGActor(nn.Module):

    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh() )

    def forward(self, x):
       return self.net(x)

class DDPGCritic(nn.Module):       
    
    def __init__(self, obs_size, act_size):

        super(DDPGCritic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            )


        self.out_net = nn.Sequential(
                       nn.Linear(400 + act_size, 300),
                       nn.ReLU(),
                       nn.Linear(300, 1)
                       )
    
    def forward(self, x, a):
        obs = self.obs_net(x)           
        return self.out_net(torch.cat([obs, a],dim=1))

def main(Models: list = [Toy_Model_NE_1.copy(), Toy_Model_NE_2.copy()], max_time: int = 100, Dil_Rate: float = 0.000000001, alpha: float = 0.01, Starting_Q: str = "FBA",Value_Update_Window=10):
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
        m.value=DDPGCritic(len(m.observables),len(m.actions))
        m.R=0
        m.optimizer_policy=optim.Adam(params=m.policy.parameters(), lr=0.01)
        m.optimizer_value=optim.Adam(params=m.value.parameters(), lr=0.01)
        m.Net_Obj=nn.MSELoss()
        m.epsilon=0.05
        m.buffer=Buffer(Value_Update_Window)
        m.alpha=0.01
        
    ### I Assume that the environment states are all observable. Env states will be stochastic
    Params["Env_States"]=Models[0].observables
    Params["Env_States_Initial_Ranges"]=[[0.1,0.1+0.00000001],[0.1,0.1+0.00000001],[100,100+0.00001],[0.001,0.001+0.00000000001],[0.001,0.001+0.00000000001]]
    for i in range(len(Models)):
        Init_C[i] = 0.001
        #Models[i].solver = "cplex"
    # writer = SummaryWriter(comment="-DeepRLDFBA_NECOM")
    Outer_Counter = 0


    for c in range(NUMBER_OF_BATCHES):
        for m in Models:
            m.epsilon=1/(1+np.exp(c/20))
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
    
    # Time=datetime.datetime.now().strftime("%d_%m_%Y.%H_%M_%S")
    # Results_Dir=os.path.join(Main_dir,"Outputs",str(Time))
    # os.mkdir(Results_Dir)
    # with open(os.path.join(Results_Dir,"Models.pkl"),'wb') as f:
    #     pickle.dump(Models,f)


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

            M.a=M.policy(torch.FloatTensor([C[M.observables]])).detach().numpy()[0]*(1-M.epsilon)+np.random.uniform(low=-5, high=5,size=len(M.actions))*M.epsilon
        
        else:

            M.a=M.policy(torch.FloatTensor([C[M.observables]])).detach().numpy()[0]
        
        for index,item in enumerate(Mapping_Dict["Ex_sp"]):
            if Mapping_Dict['Mapping_Matrix'][index,i]!=-1:
                M.reactions[Mapping_Dict['Mapping_Matrix'][index,i]].upper_bound=20
                M.reactions[Mapping_Dict['Mapping_Matrix'][index,i]].lower_bound=-General_Uptake_Kinetics(C[index+len(Models)])
                
            
        for index,flux in enumerate(M.actions):
            M.a[index]=Flux_Clipper(M.reactions[M.actions[index]].lower_bound,M.a[index],M.reactions[M.actions[index]].upper_bound)
            M.reactions[M.actions[index]].lower_bound=M.a[index]
            M.reactions[M.actions[index]].upper_bound=M.a[index]

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


    for m in Models:
        m.buffer.update_queue((C[m.observables],m.a,m.reward))
        if Counter>0 and Counter%len(m.buffer.queue)==0:
            # TD_Error=[]
            States=[]
            Actions=[]
            TD_q=[]
            for obs in range(len(m.buffer.queue)-1,0,-1):
                States.append(m.buffer.queue[obs][0])
                Actions.append(m.buffer.queue[obs][1])
                TD_q.append(m.buffer.queue[obs][2]+m.value(torch.FloatTensor([m.buffer.queue[obs-1][0]]),torch.FloatTensor([m.buffer.queue[obs-1][1]])))
                # TD_Error.append(m.buffer.queue[obs][2]-m.R+m.value(torch.FloatTensor(m.buffer.queue[obs-1][0]),torch.FloatTensor(m.buffer.queue[obs-1][1]))-m.value(torch.FloatTensor(States[-1]),torch.FloatTensor((Actions[-1]))))
                # m.R+=m.alpha*TD_Error[-1]
            m.value.zero_grad()
            q_v = m.value(torch.FloatTensor(States), torch.FloatTensor(Actions))
            TD_q_v=torch.FloatTensor(TD_q)
            m.optimizer_value.zero_grad()
            loss_c=F.mse_loss(q_v,TD_q_v.detach())
            loss_c.backward()
            m.optimizer_value.step()
            m.optimizer_policy.zero_grad()
            cur_actions_v = m.policy(torch.FloatTensor(States))
            actor_loss_v = -m.value(torch.FloatTensor(States), cur_actions_v)
            actor_loss_v = actor_loss_v.mean()
            actor_loss_v.backward()
            m.optimizer_policy.step()


    dCdt += np.array(Params["Dilution_Rate"])*(Params["Inlet_C"]-C)
    
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
    return 10*(Compound/(Compound+20))





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
        dFBA(Models, Mapping_Dict, Init_C, Params, t_span, dt=dt)
    
        for mod in Models:
            print(f"{BATCH} - {mod.NAME} earned {mod.reward} during this episode!")
    





def Flux_Clipper(Min,Number,Max):
    return(min(max(Min,Number),Max))
    

if __name__ == "__main__":
    # Init_Pols=[]
    # for i in range(2):
    #     Init_Pols.append(os.path.join(Main_dir,"Outputs","Agent_"+str(i)+"_3900.pkl"))

    # cProfile.run("","Profile")
    main([Toy_Model_NE_1.copy(),Toy_Model_NE_2.copy()])

