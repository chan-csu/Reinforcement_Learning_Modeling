
# Script for running Community Dynamic Flux Balance Analysis (CDFBA)
# Written by: Parsa Ghadermazi

import numpy as np
import cobra
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import cProfile
import pstats

# import plotext as plx

Main_dir = os.path.dirname(os.path.abspath(__file__))


def main(Number_of_Models: int = 2, max_time: int = 100, Dil_Rate: float = 0.01):
    """
    This is the main function for running dFBA.
    The main requrement for working properly is
    that the models use the same notation for the
    same reactions.


    """

    Models = []
    Base_Model = cobra.io.read_sbml_model(
        os.path.join(Main_dir, 'IJO1366_AP.xml'))
    [Models.append(Base_Model.copy()) for i in range(Number_of_Models)]

    for i in range(Number_of_Models):
        Models[i].name = "Ecoli_"+str(i+1)
        Models[i].solver.objective.name = "_pfba_objective"

    Mapping_Dict = Build_Mapping_Matrix(Models)
    Init_C = np.ones((Models.__len__()+Mapping_Dict["Ex_sp"].__len__()+1,))
    Inlet_C = np.zeros((Models.__len__()+Mapping_Dict["Ex_sp"].__len__()+1,))

    # The Params are the main part to change from problem to problem

    Params = {
        "Dilution_Rate": Dil_Rate,
        "Glucose_Index": Mapping_Dict["Ex_sp"].index("EX_glc__D(e)")+Models.__len__(),
        "Starch_Index": Mapping_Dict["Ex_sp"].__len__()+Models.__len__(),
        "Amylase_Ind": Mapping_Dict["Ex_sp"].index("EX_amylase(e)")+Models.__len__(),
        "Inlet_C": Inlet_C,
        "Model_Glc_Conc_Index": [Models[i].reactions.index("EX_glc__D(e)") for i in range(Number_of_Models)],
        "Model_Amylase_Conc_Index": [Models[i].reactions.index("EX_amylase(e)") for i in range(Number_of_Models)],
        "Num_Glucose_States": 10,
        "Num_Starch_States": 10,
        "Num_Amylase_States": 10,
        "Glc_Max_C": 100,
        "Starch_Max_C": 10,
        "Amylase_Max_C": 10,


    }

    Init_C[[Params["Glucose_Index"],
            Params["Starch_Index"], Params["Amylase_Ind"]]] = [100, 1, 1]
    Inlet_C[Params["Starch_Index"]] = 1
    Params["Inlet_C"] = Inlet_C

    for i in range(Number_of_Models):
        Init_C[i] = 0.001
        Models[i].solver="cplex"

    # Policy initialization
    # Initial Policy is set to a random policy
    Init_Policy_Dict = {}
    States = [(i, j, k) for i in range(Params["Num_Glucose_States"]) for j in range(
        Params["Num_Starch_States"]) for k in range(Params["Num_Amylase_States"])]
    for state in States:
        Init_Policy_Dict[state] = random.choice(range(9))

    for i in range(Number_of_Models):
        Models[i].Policy = Policy_Deterministic(Init_Policy_Dict)

    Returns=Generate_Episodes_With_State(dFBA, Params, Init_C, Models, Mapping_Dict, Num_Episodes=2, Gamma=1)

    print(Returns)
    ############################
    # Saving the policy with pickle place holder
    ############################


def dFBA(Models, Mapping_Dict, Init_C, Params,dt=0.1):
    """
    This function calculates the concentration of each species
    Models is a list of COBRA Model objects
    Mapping_Dict is a dictionary of dictionaries
    """



    ##############################################################
    # Initializing the ODE Solver
    ##############################################################
    t = np.arange(0, 100, dt)
    ##############################################################
    # Solving the ODE
    ##############################################################

    States=[0]*t.__len__()
    Actions=[[0]*Models.__len__()]*t.__len__()

    sol, t = odeFwdEuler(ODE_System, Init_C, 0.1,  Params,
                         [0, 10], Models, Mapping_Dict,States,Actions)
    return sol, t,States,Actions


def ODE_System(C, t, Models, Mapping_Dict, Params,States,Actions,Integrator_Counter):
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
    dCdt = np.zeros(C.shape)
    Sols = [0]*Models.__len__()

    for i in range(Models.__len__()):
        Models[i].State = Descretize_Concs(
            C[Params["Glucose_Index"]], C[Params["Starch_Index"]], C[Params["Amylase_Ind"]], Params)

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
        if t == 0:
            Models[i].reactions[Params["Model_Amylase_Conc_Index"]
                                [i]].lower_bound = (lambda x, a: a*x/100)(Models[i].InitAction, 1)
            Actions[0][i]=Models[i].InitAction
        else:
            Models[i].reactions[Params["Model_Amylase_Conc_Index"]
                            [i]].lower_bound = (lambda x, a: a*x/100)(Models[i].Policy.get_action(Models[i].State), 5)
        Sols[i] = Models[i].optimize()
        dCdt[i] += Sols[i].objective_value*C[i]


    for i in range(Mapping_Dict["Mapping_Matrix"].shape[0]):
        for j in range(Models.__len__()):
            if Mapping_Dict["Mapping_Matrix"][i, j] != -1:
                dCdt[i+Models.__len__()] += Sols[j].fluxes.iloc[Mapping_Dict["Mapping_Matrix"]
                                                                [i, j]]*C[j]
            if Mapping_Dict["Ex_sp"][i] == "EX_glc__D(e)":
                dCdt[i+Models.__len__()] += Starch_Degradation_Kinetics(
                    C[Params["Amylase_Ind"]], C[Params["Starch_Index"]])

    dCdt[Params["Starch_Index"]] = - \
        Starch_Degradation_Kinetics(
            C[Params["Amylase_Ind"]], C[Params["Starch_Index"]])

    dCdt += np.array(Params["Dilution_Rate"])*(Params["Inlet_C"]-C)
    print(t)
    for i in range(Models.__len__()):
        if t!=0:
            Actions[Integrator_Counter][i]=Models[i].Policy.Policy[Models[i].State]
    States[Integrator_Counter]=Models[0].State
    return dCdt


def Find_Optimal_Policy(Models, Mapping_Dict, ICs, Params):
    pass


def Generate_Episode():
    pass


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


def Starch_Degradation_Kinetics(a_Amylase: float, Starch: float, Model="", k: float = 0.001):
    """
    This function calculates the rate of degradation of starch
    a_Amylase Unit: mmol
    Starch Unit: mg

    """

    return a_Amylase*Starch*k


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


def Descretize_Concs(Glucose: float, Starch: float, Amylase: float, Params):
    """
    This function calculates the state of the reactor


    """

    Glucose1 = math.floor(Glucose/Params["Glc_Max_C"]*10)
    Starch1 = math.floor(Starch/Params["Starch_Max_C"]*10)
    Amylase1 = math.floor(Amylase/Params["Amylase_Max_C"]*10)
    if Glucose >= Params["Glc_Max_C"]:
        Glucose1 = 9
    if Starch >= Params["Starch_Max_C"]:
        Starch1 = 9
    if Amylase >= Params["Amylase_Max_C"]:
        Amylase1 = 9

    return (Glucose1, Starch1, Amylase1)


def odeFwdEuler(ODE_Function, ICs, dt, Params, t_span, Models, Mapping_Dict,States,Actions):
    Integrator_Counter=0
    t = np.arange(t_span[0], t_span[1], dt)
    sol = np.zeros((len(t), len(ICs)))
    sol[0] = ICs
    for i in range(1, len(t)):
        sol[i] = sol[i-1] + \
            ODE_Function(sol[i-1], t[i-1], Models, Mapping_Dict, Params,States,Actions,Integrator_Counter)*dt
        Integrator_Counter+=1
    return sol, t


def Generate_Episodes_With_State(dFBA, Params, Init_C, Models, Mapping_Dict, Num_Episodes=2, Gamma=1):

    Returns = {}
    for i in range(Models.__len__()):
        Returns[i] = []

    for k in range(Num_Episodes):
        Init_C[[Params["Glucose_Index"],
                Params["Starch_Index"],
                Params["Amylase_Ind"]]] = [random.uniform(0, Params["Glc_Max_C"]*1.1),
                                           random.uniform(0, Params["Starch_Max_C"]*1.1),
                                           random.uniform(0, Params["Amylase_Max_C"]*1.1)]

        for i in range(Models.__len__()):
            Models[i].InitAction = random.choice(
                range(Params["Num_Amylase_States"]))

        C, _,States,Actions = dFBA(Models, Mapping_Dict, Init_C, Params,dt=0.1)
        for j in range(Models.__len__()):
            Returns[j].append(np.average(C[1:, j]))  # No GAMMA for now

    return Returns


if __name__ == "__main__":
    main()

 # SCRATCH PAPER
#  Solution=scipy.integrate.solve_ivp(ODE_Sys, (0, 10),  ICs, t_eval=np.linspace(
#     0, 10, num=1000), method='Radau', args=[Params])
