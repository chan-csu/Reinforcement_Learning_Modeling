
# Script for running Community Dynamic Flux Balance Analysis (CDFBA)
# Written by: Parsa Ghadermazi
from email import iterators
import numpy as np
import cobra
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import cProfile
import pstats
import concurrent.futures
import multiprocessing
import pickle
import itertools
# import cplex
from ToyModel import ToyModel
CORES = multiprocessing.cpu_count()

# import plotext as plx

Main_dir = os.path.dirname(os.path.abspath(__file__))


def main(Models: list = [ToyModel.copy(), ToyModel.copy()],Pol_Cases=None ,Test_Conditions=None,max_time: int = 1000, Dil_Rate: float = 0.1, alpha: float = 0.1, Starting_Policy: str = "Random"):
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
        "Glucose_Max_C": 100,
        "Starch_Max_C": 10,
        "Amylase_Max_C": 1,
        "Agent_Max_C": 10,
        "alpha": alpha,
        "STATES":("Glucose","Starch","Agent")


    }

    Init_C[[Params["Glucose_Index"],
            Params["Starch_Index"], *Params["Agents_Index"]]] = [Test_Conditions["Glucose"],Test_Conditions["Starch"],*Test_Conditions["Agents"]]
    Inlet_C[Params["Starch_Index"]] = 10
    Params["Inlet_C"] = Inlet_C

    # for i in range(Number_of_Models):
    #     Init_C[i] = 0.001
    #     # Models[i].solver = "cplex"

    # ----------------------------------------------------------------------------

    # Policy initialization ###--------------------------------------------------
    # Initial Policy is set to a random policy

    States = [(i, j, k) for i in range(Params["Num_Glucose_States"]) for j in range(
        Params["Num_Starch_States"]) for k in range(Params["Number_of_Agent_States"])]

    fig,ax=plt.subplots(Pol_Cases[Models[0]._name].__len__(),2)
    for i in range(Pol_Cases[Models[0]._name].__len__()):
        for j in range(Number_of_Models):
            with open(Pol_Cases[Models[j]._name][i], "rb") as f:
                Models[j].Policy = Policy_Deterministic(pickle.load(f).copy())






        C, t = Generate_Episodes_With_State(dFBA, States, Params, Init_C, Models, Mapping_Dict, t_span=[
            0, max_time], dt=0.1)
        ax[i,0].plot(t,C[:,0:Models.__len__()])
        ax[i,0].legend([Mod._name for Mod in Models])
        Iter=Pol_Cases[Models[j]._name][i].split("_")[-1].split(".")[0]
        ax[i,0].set_title(f"Concentration Profile of Agents: Iter {Iter}")
        ax[i,1].set_title(f"Metabolite Concentrations")
        ax[i,1].plot(t,C[:,-2])



    # plt.tight_layout()
    plt.show()
    ############################
    # Saving the policy with pickle place holder once in a while
    ############################
    


def dFBA(Models, Mapping_Dict, Init_C, Params, t_span, dt=0.05):
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

    for i in range(Models.__len__()):
        Models[i].State = Descretize_Concs(Params,
            (C[Params["Glucose_Index"]], C[Params["Starch_Index"]], *C[Params["Agents_Index"]] ))

    for j in range(Mapping_Dict["Mapping_Matrix"].shape[0]):
        for i in range(Models.__len__()):
            if Mapping_Dict["Mapping_Matrix"][j, i] != Params["Model_Glc_Conc_Index"][i]:
                if Mapping_Dict["Mapping_Matrix"][j, i] != Params["Model_Amylase_Conc_Index"][i]:
                    pass
                    # Models[i].reactions[Mapping_Dict["Mapping_Matrix"][j, i]
                    #                     ].lower_bound = - General_Uptake_Kinetics(C[j+Models.__len__()])
            else:
                Models[i].reactions[Mapping_Dict["Mapping_Matrix"][j, i]
                                    ].lower_bound = - Glucose_Uptake_Kinetics(C[j+Models.__len__()])
    for i in range(Models.__len__()):


        Models[i].reactions[Params["Model_Amylase_Conc_Index"]
                                [i]].lower_bound = (lambda x, a: a*x)(Models[i].Policy.get_action(Models[i].State), 1)
        Sols[i] = Models[i].optimize()
        if Sols[i].status == 'infeasible':
            dCdt[i] = 0
        elif Sols[i].status == "optimal":
            dCdt[i] += Sols[i].objective_value*C[i]
        else:
            print("Flag")

    for i in range(Mapping_Dict["Mapping_Matrix"].shape[0]):
        for j in range(Models.__len__()):
            if Mapping_Dict["Mapping_Matrix"][i, j] != -1:
                if Sols[j].status == 'infeasible':
                    dCdt[i] = 0
                else:
                    dCdt[i+Models.__len__()] += Sols[j].fluxes.iloc[Mapping_Dict["Mapping_Matrix"]
                                                                    [i, j]]*C[j]
            if Mapping_Dict["Ex_sp"][i] == "Glc_Ex":
                
                if Sols[j].status == 'infeasible':
                    pass
                
                else:

                    dCdt[i+Models.__len__()] +=Sols[j].fluxes.iloc[Mapping_Dict["Mapping_Matrix"][i, j]]*C[j]
        
    dCdt[Params["Glucose_Index"]] += Starch_Degradation_Kinetics(
                        C[Params["Amylase_Ind"]], C[Params["Starch_Index"]])*10                                                                                    

    dCdt[Params["Starch_Index"]] = - \
        Starch_Degradation_Kinetics(C[Params["Amylase_Ind"]], C[Params["Starch_Index"]])/100

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


def Descretize_Concs(Params,State_Concs):
    """
    This function calculates the state of the reactor


    """
    States=Params["STATES"]
    Descritized=[]
    for i,s in enumerate(States):
        Descritized.append(math.floor(State_Concs[i]/Params[s+"_Max_C"]*10)) if State_Concs[i]<Params[s+"_Max_C"] else Descritized.append(9)
    
    return tuple(Descritized)


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


def Generate_Episodes_With_State(dFBA, States, Params, Init_C, Models, Mapping_Dict, t_span=[0, 100], dt=0.1):
    Returns_Totall = {}
    for M in range(Models.__len__()):
        Returns_Totall[M] = {}
        for state in States:
            for action in range(Params["Num_Amylase_States"]):
                Returns_Totall[M][(state, action)] = []


    for i in range(Models.__len__()):

        Models[i].Init_State = Descretize_Concs(Params,(Init_C[Params["Glucose_Index"]], Init_C[Params["Starch_Index"]], *Init_C[Params["Agents_Index"]]))

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
    
    
    
    ### Defining the conditions for testing###
    
    Test_Condition={"Glucose":90,
                    "Starch":10,
                    "Agents":[0.001]}
        
    ##########################################

    ###  Agent and Policy Initialization  ###
    Agent_Names=["Agent_0"]
    Case_Dir=os.path.join(Main_dir,"Cases","Case_4")
    Policies={}
    Pols=[DIR for DIR in os.listdir(Case_Dir) if os.path.isfile(os.path.join(Case_Dir,DIR)) and "Agent" in DIR]
    iters=list(set([int(POL.split("_")[-1].split(".")[0]) for POL in Pols]))
    iters.sort()
    Agents=[]
    for i in range(Agent_Names.__len__()):
        Policies[Agent_Names[i]]=[]
        for iter in iters:
            Policies[Agent_Names[i]].append(os.path.join(Case_Dir,Agent_Names[i]+"_"+str(iter)+".pkl"))
        
        Agents.append(ToyModel.copy())
        Agents[i]._name=Agent_Names[i]







    main(Models=Agents,Pol_Cases=Policies,Test_Conditions=Test_Condition)
