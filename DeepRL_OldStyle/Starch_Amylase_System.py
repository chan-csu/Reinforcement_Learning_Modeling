
# Script for running Community Dynamic Flux Balance Analysis (CDFBA)
# Written by: Parsa Ghadermazi

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
from ToyModel import ToyModel_Starch_Amylase as ToyModel_SA
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import ray
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter

Scaler=StandardScaler()

NUMBER_OF_BATCHES=200
BATCH_SIZE=16
HIDDEN_SIZE=20
PERCENTILE=70
CORES = multiprocessing.cpu_count()
Main_dir = os.path.dirname(os.path.abspath(__file__))
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),   
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, n_actions),
            
        )

    def forward(self, x):
        return self.net(x)


def main(Models: list = [ToyModel.copy(), ToyModel.copy()], max_time: int = 100, Dil_Rate: float = 0.1, alpha: float = 0.01, Starting_Q: str = "FBA"):
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
    Number_of_Models = Models.__len__()
    for i in range(Number_of_Models):
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
    Init_C = np.ones((Models.__len__()+Mapping_Dict["Ex_sp"].__len__()+1,))
    Inlet_C = np.zeros((Models.__len__()+Mapping_Dict["Ex_sp"].__len__()+1,))

    # The Params are the main part to change from problem to problem

    Params = {
        "Dilution_Rate": Dil_Rate,
        "Glucose_Index": Mapping_Dict["Ex_sp"].index("Glc_Ex")+Models.__len__(),
        "Starch_Index": Mapping_Dict["Ex_sp"].__len__()+Models.__len__(),
        "Amylase_Ind": Mapping_Dict["Ex_sp"].index("Amylase_Ex")+Models.__len__(),
        "Inlet_C": Inlet_C,
        "Model_Glc_Conc_Index": [Models[i].reactions.index("Glc_Ex") for i in range(Number_of_Models)],
        "Model_Amylase_Conc_Index": [Models[i].reactions.index("Amylase_Ex") for i in range(Number_of_Models)],
        "Agents_Index": [i for i in range(Number_of_Models)],
        "Num_Glucose_States": 10,
        "Num_Starch_States": 10,
        "Num_Amylase_States": 10,
        "Number_of_Agent_States": 10,
        "Glucose_Max_C": 200,
        "Starch_Max_C": 10,
        "Amylase_Max_C": 1,
        "Agent_Max_C": 1,
        "alpha": alpha,
        "beta":alpha
    }

    Params["State_Inds"]=[ind for ind in Params["Agents_Index"]]
    Params["State_Inds"].append(Params["Glucose_Index"])
    Params["State_Inds"].append(Params["Starch_Index"])
    
    for ind,m in enumerate(Models):
        m.observation=Params["State_Inds"].copy()
        m.actions=[Params["Model_Amylase_Conc_Index"][ind]]
        m.Policy=Net(len(m.observation), HIDDEN_SIZE, len(m.actions))
        m.optimizer=optim.Adam(params=m.Policy.parameters(), lr=0.01)
        m.Net_Obj=nn.MSELoss()
        m.epsilon=0.01
    
    Inlet_C[Params["Starch_Index"]] = 10
    Params["Inlet_C"] = Inlet_C
    
    for i in range(Number_of_Models):
        Init_C[i] = 0.001
        #Models[i].solver = "cplex"
    writer = SummaryWriter(comment="-DeepRLDFBA")
    Outer_Counter = 0


    for c in range(NUMBER_OF_BATCHES):
        
        Batch_Out=Generate_Batch(dFBA, Params, Init_C, Models, Mapping_Dict,Batch_Size=BATCH_SIZE)
        Batch_Out=list(map(list, zip(*Batch_Out)))
        for index,Model in enumerate(Models):
            obs_v, acts_v, reward_b, reward_m=filter_batch(Batch_Out[index], PERCENTILE)
            Model.optimizer.zero_grad()
            action_scores_v = Model.Policy(obs_v)
            loss_v = Model.Net_Obj(action_scores_v, acts_v)
            loss_v.backward()
            Model.optimizer.step()
            print(f"{Model.NAME}")
            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (c, loss_v.item(), reward_m, reward_b))

            writer.add_scalar(f"{Model.NAME} reward_mean", reward_m, c)





@ray.remote
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


def ODE_System(C, t, Models, Mapping_Dict, Params, dt):
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

            M.a=random.random()*30
        
        else:

            M.a=M.Policy(torch.FloatTensor([C[Params["State_Inds"]]])).item()

        M.reactions[Params["Model_Amylase_Conc_Index"]
                                [i]].lower_bound = (lambda x, a: a*x)(M.a, 1)
        
        M.reactions[Params["Model_Glc_Conc_Index"][i]
                                    ].lower_bound = - Glucose_Uptake_Kinetics(C[Params["Glucose_Index"]])
        
        Sols[i] = Models[i].optimize()

        if Sols[i].status == 'infeasible':
            Models[i].reward= -10
            dCdt[i] = 0

        else:
            dCdt[i] += Sols[i].objective_value*C[i]
            Models[i].reward =Sols[i].objective_value



    ### Writing the balance equations

    for i in range(Mapping_Dict["Mapping_Matrix"].shape[0]):
        for j in range(Models.__len__()):
            if Mapping_Dict["Mapping_Matrix"][i, j] != -1:
                if Sols[j].status == 'infeasible':
                    dCdt[i] = 0
                else:
                    dCdt[i+Models.__len__()] += Sols[j].fluxes.iloc[Mapping_Dict["Mapping_Matrix"]
                                                                    [i, j]]*C[j]

    dCdt[Params["Glucose_Index"]] += Starch_Degradation_Kinetics(
                        C[Params["Amylase_Ind"]], C[Params["Starch_Index"]])*10

    dCdt[Params["Starch_Index"]] = - \
        Starch_Degradation_Kinetics(
            C[Params["Amylase_Ind"]], C[Params["Starch_Index"]])/100

    for m in Models:
        m.episode_reward += m.reward
        m.episode_steps.append(EpisodeStep(observation=C[Params["State_Inds"]], action=m.a))
    
    dCdt += np.array(Params["Dilution_Rate"])*(Params["Inlet_C"]-C)
    
    return dCdt


def Build_Mapping_Matrix(Models):
    """
    Given a list of COBRA model objects, this function will build a mapping matrix

    """

    Ex_sp = []
    for model in Models:
        for Ex_rxn in model.exchanges:
            if Ex_rxn.id not in Ex_sp:
                Ex_sp.append(Ex_rxn.id)
    Mapping_Matrix = np.zeros((len(Ex_sp), len(Models)), dtype=int)
    for i, id in enumerate(Ex_sp):
        for j, model in enumerate(Models):
            if id in model.reactions:
                Mapping_Matrix[i, j] = model.reactions.index(id)
            else:
                Mapping_Matrix[i, j] = -1
    return {"Ex_sp": Ex_sp, "Mapping_Matrix": Mapping_Matrix, "Models": Models}


class Policy_Deterministic:
    def __init__(self, Policy):
        self.Policy = Policy

    def get_action(self, state):
        return self.Policy[state]





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
                         Params, dt)*dt
        Integrator_Counter += 1
    return sol, t


def Generate_Batch(dFBA, Params, Init_C, Models, Mapping_Dict, Batch_Size=10,t_span=[0, 100], dt=0.1):


    Init_C[[Params["Glucose_Index"],
            Params["Starch_Index"]]] = [random.uniform(Params["Glucose_Max_C"]*0.5, Params["Glucose_Max_C"]*1.5),
                                       random.uniform(
                                           Params["Starch_Max_C"]*0.5, Params["Starch_Max_C"]*1.5)]
    
    for agent_ind in Params["Agents_Index"]:
        Init_C[agent_ind]=random.uniform(Params["Agent_Max_C"]*0.5, Params["Agent_Max_C"]*1.5)





    
    Batch_Episodes=[]
    for _ in range(Batch_Size):
        Batch_Episodes.append(dFBA.remote(Models, Mapping_Dict, Init_C, Params, t_span, dt=dt))

    return(ray.get(Batch_Episodes))
    



def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.FloatTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean




if __name__ == "__main__":
    # Init_Pols=[]
    # for i in range(2):
    #     Init_Pols.append(os.path.join(Main_dir,"Outputs","Agent_"+str(i)+"_3900.pkl"))

    # cProfile.run("","Profile")
    ray.init()
    main([ToyModel_SA.copy()])

