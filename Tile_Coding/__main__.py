
# Script for running Community Dynamic Flux Balance Analysis (CDFBA)
# Written by: Parsa Ghadermazi

import numpy as np
import cobra
import os
import random
import matplotlib.pyplot as plt
import plotext as plx
import numpy as np
import math
import cProfile
import pstats
import concurrent.futures
import multiprocessing
import pickle
import itertools
import pandas
#import cplex
from ToyModel import ToyModel
from dataclasses import dataclass,field
CORES = multiprocessing.cpu_count()

Main_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Feature_Vector:
    Number_of_tiles: int =10
    Number_of_Tiling:int =10
    State_Dimensions: int=2
    State_Ranges:list= field(default_factory=lambda:[[2,2],[1,1]])
    Number_of_actions:int=3
    
    def __post_init__(self):
        
        self._Empty_Feature_vect=np.zeros((self.State_Dimensions,self.Number_of_Tiling,self.Number_of_tiles,self.Number_of_actions))
        self.Shift_Vect=[(i[1]-i[0])/self.Number_of_tiles/self.Number_of_Tiling for i in self.State_Ranges]
        self._Base_Shift_Vect=np.array([2*i+1 for i in range(self.Number_of_Tiling)]) 
        Temp=np.empty((self.State_Dimensions,self.Number_of_Tiling)).astype(np.ndarray)
        for i in range(self.State_Dimensions):
            for j in range(self.Number_of_Tiling):
                Temp[i,j]=np.linspace(self.State_Ranges[i][0]+self._Base_Shift_Vect[j]*(i+1)*self.Shift_Vect[i],self.State_Ranges[i][1]+self._Base_Shift_Vect[j]*(i+1)*self.Shift_Vect[i],num=self.Number_of_tiles-1)

        self.bin=Temp

    def Get_Feature_Vector(self,State_Action):
        Index=np.zeros((self.State_Dimensions*self.Number_of_Tiling,4),dtype=int)
        Index[...,-1]=State_Action[1]
        Index[...,1]=np.tile(np.arange(self.Number_of_Tiling),self.State_Dimensions)
        Index[...,0]=np.repeat(range(self.State_Dimensions),self.Number_of_Tiling)
        Index[...,2]=np.array([np.digitize(State_Action[0][i],self.bin[i,j]) for j in range(self.Number_of_Tiling) for i in range(self.State_Dimensions)])
        
        return tuple(Index.T)


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
        "alpha": alpha
    }

    Params["State_Inds"]=[0,1,Params["Glucose_Index"],Params["Starch_Index"]]
    Ranges=[[0,1] for i in range(Number_of_Models)]
    Ranges.append([0,Params["Glucose_Max_C"]])
    Ranges.append([0,Params["Starch_Max_C"]])
    
    for m in Models:
        m.Features=Feature_Vector(10,10,Number_of_Models+2,Ranges,Params["Num_Amylase_States"])
        m.alpha=Params["alpha"]
        m.W=m.Features._Empty_Feature_vect.copy()
        m.Actions=range(Params["Num_Amylase_States"])
        m.epsilon=0.1



    # Init_C[[Params["Glucose_Index"],
    #         Params["Starch_Index"], Params["Amylase_Ind"]]] = [100, 1, 1]
    Inlet_C[Params["Starch_Index"]] = 10
    Params["Inlet_C"] = Inlet_C
    F={}
    for i in range(Number_of_Models):
        Init_C[i] = 0.001
        F[Models[i].NAME]=[]
        F[Models[i].NAME+"_abs_W_max"]=[]
        #Models[i].solver = "cplex"

    # ----------------------------------------------------------------------------

    # Policy initialization ###--------------------------------------------------
    # Initial Policy is set to a random policy


    if not os.path.exists(os.path.join(Main_dir, "Outputs")):
        os.mkdir(os.path.join(Main_dir, "Outputs"))

    for i in range(Number_of_Models):
        if Starting_Q == "FBA":
            pass 
        else:
            
            with open(Starting_Q[i], "rb") as f:
                Models[i].W= pickle.load(f)
             
        
        with open(os.path.join(Main_dir, "Outputs", Models[i].NAME+"_0.pkl"), "wb") as f:
            pickle.dump(Models[i].W, f)

    Outer_Counter = 0
    # Q initalization ###------------------------------------------------------------
    while True:
        
        C, t = Generate_Episodes_With_State(dFBA, Params, Init_C, Models, Mapping_Dict, t_span=[
            0, max_time], dt=0.1)
        
        for i in range(Models.__len__()):
            F[Models[i].NAME].append(np.sum(Models[i].f_values))
            F[Models[i].NAME+"_abs_W_max"].append(np.max(np.abs(Models[i].W)))
        pandas.DataFrame(F).to_csv(os.path.join(Main_dir,"F.csv"))
        # for i in range(Models.__len__()):
        #          F[i].append(np.sum(Models[i].f_values))
        #          plt.plot(range(len(F[i])),F[i])
        print(f"Iter: {Outer_Counter}")
        print(f"End_Concs: {list([C[-1,i] for i in range(Number_of_Models)])}")
        print(
            f"Returns: {list([np.sum(Models[i].f_values) for i in range(Number_of_Models)])}")
    ############################
    # Saving the policy with pickle place holder once in a while
    ############################
        if Outer_Counter % 1000 == 0:
            for i in range(Number_of_Models):
                with open(os.path.join(Main_dir, "Outputs", Models[i].NAME+"_"+str(Outer_Counter)+".pkl"), "wb") as f:
                    pickle.dump(Models[i].W, f)
       
        Outer_Counter += 1

def dFBA(Models, Mapping_Dict, Init_C, Params, t_span, dt=0.1):
    """
    This function calculates the concentration of each species
    Models is a list of COBRA Model objects
    Mapping_Dict is a dictionary of dictionaries
    """
    for i in range(Models.__len__()):
        Models[i].f_values=[] 
    ##############################################################
    # Initializing the ODE Solver
    ##############################################################
    t = np.arange(t_span[0], t_span[1], dt)
    ##############################################################
    # Solving the ODE
    ##############################################################

    sol, t = odeFwdEuler(ODE_System, Init_C, dt,  Params,
                         t_span, Models, Mapping_Dict)
    return sol, t


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
        Temp_O=np.array([np.sum(M.W[M.Features.Get_Feature_Vector((C[Params["State_Inds"]],idx))]) for idx in M.Actions])
        if np.random.random()<M.epsilon:
            M.a=np.random.choice(M.Actions)
        else:
            M.a=np.argmax(Temp_O)
        M.q=Temp_O[M.a]

        M.reactions[Params["Model_Amylase_Conc_Index"]
                                [i]].lower_bound = (lambda x, a: a*x)(M.a, 1)
        
        M.reactions[Params["Model_Glc_Conc_Index"][i]
                                    ].lower_bound = - Glucose_Uptake_Kinetics(C[Params["Glucose_Index"]])
        
        Sols[i] = Models[i].optimize()

        if Sols[i].status == 'infeasible':
            Models[i].f_values.append(-10)
            dCdt[i] = 0

        else:
            dCdt[i] += Sols[i].objective_value*C[i]
            Models[i].f_values.append(Sols[i].objective_value)



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

    dCdt += np.array(Params["Dilution_Rate"])*(Params["Inlet_C"]-C)
    Next_C = C[Params["State_Inds"]]+dCdt[Params["State_Inds"]]*dt
    Next_C[Next_C < 0] = 0


    for z in Models:

        qp=np.array([np.sum(z.W[z.Features.Get_Feature_Vector((Next_C,idx))]) for idx in z.Actions]).max()
 
        z.W[z.Features.Get_Feature_Vector((C[Params["State_Inds"]], z.a))] += Params['alpha']*(z.f_values[-1]+qp-z.q)

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


class Policy_General:
    """
    Any policy would be an instance of this class
    Given a state it will retrun a dictionary including the probability distribution o
    each action
    """

    def __init__(self, Porb_Dist_Dict):
        self.Policy = Porb_Dist_Dict

    def get_action(self, state):
        Actions = [(action, self.Policy[state][action])
                   for action in self.Policy[state].keys()]
        np.random.choice(Actions, p=[action[1] for action in Actions], k=1)


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


def Generate_Episodes_With_State(dFBA, Params, Init_C, Models, Mapping_Dict, t_span=[0, 100], dt=0.1):


    Init_C[[Params["Glucose_Index"],
            Params["Starch_Index"],
            *Params["Agents_Index"]]] = [random.uniform(Params["Glucose_Max_C"]*0.5, Params["Glucose_Max_C"]*1.5),
                                       random.uniform(
                                           Params["Starch_Max_C"]*0.5, Params["Starch_Max_C"]*1.5),
                                       random.uniform(Params["Agent_Max_C"]*0.5, Params["Agent_Max_C"]*1.5),random.uniform(Params["Agent_Max_C"]*0.5, Params["Agent_Max_C"]*1.5)]

    C, t = dFBA(
        Models, Mapping_Dict, Init_C, Params, t_span, dt=dt)

    return C, t

    # Here we can change np.sum(C[*, m]) to C[-1, m] to favor final steady state concentratins
    # Or even other smart things!
    # Leg=[]
    # plt.cla()
    # for i in range(Models.__len__()):
    #     plt.plot(t, C[:, i])
    #     Leg.append(Models[i].name)
    # plt.plot(t,C[:, Params["Glucose_Index"]])
    # plt.plot(t,C[:, Params["Starch_Index"]])
    # plt.plot(t,C[:, Params["Amylase_Ind"]])
    # Leg.append("Glucose")
    # Leg.append("Starch")
    # Leg.append("Amylase")
    # plt.legend(Leg)
    # plt.ioff()

    # plt.show()


if __name__ == "__main__":
    # Init_Pols=[]
    # for i in range(2):
    #     Init_Pols.append(os.path.join(Main_dir,"Outputs","Agent_"+str(i)+"_3900.pkl"))

    main([ToyModel.copy(),ToyModel.copy()])
