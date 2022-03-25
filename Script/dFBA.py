
# Script for running Community Dynamic Flux Balance Analysis (CDFBA)
# Written by: Parsa Ghadermazi

import numpy as np
import cobra
import os
import scipy

Main_dir = os.path.dirname(os.path.abspath(__file__))


def main(Number_of_Models: int = 2, max_time: int = 100, Dil_Rate: float = 0.1):
    """
    This is the main function for running dFBA.
    The main requrement for working properly is
    that the models use the same notation for the
    same reactions.


    """

    Models = []
    Main_dir = os.path.dirname(os.path.abspath(__file__))
    Base_Model = cobra.io.read_sbml_model(
        os.path.join(Main_dir, 'IJO1366_AP.xml'))
    [Models.append(Base_Model.copy()) for i in range(Number_of_Models)]

    for i in range(Number_of_Models):
        Models[i].name = "Ecoli_"+str(i+1)

    Mapping_Dict = Build_Mapping_Matrix(Models)
    Init_C = np.zeros((Models.__len__()+Mapping_Dict["Ex_sp"].__len__(), 1))
    Inlet_C = np.zeros((Models.__len__()+Mapping_Dict["Ex_sp"].__len__(), 1))

    Params = {
        "Dilution_Rate": Dil_Rate,
        "Glucose_Index": None,
        "Starch_Index": None,


    }

    ############################
    # Initial Policy Placeholder
    ############################
    Opt_Policy = Find_Optimal_Policy(
        Models, Mapping_Dict, ICs, Params, Initial_Policy)
    ############################
    # Saving the policy with pickle place holder
    ############################


def dFBA(Models, Mapping_Dict, Pol, Init_C, Inlet_C):
    """
    Main function for running Dynamic Flux Balance Analysis (dFBA)
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


def ODE_System(t, C, Models, Mapping_Dict, Params):
    """
    This function calculates the differential equations for the system
    """
    dCdt = np.zeros(C.shape)
    for j in range(Mapping_Dict["Mapping_Matrix"].shape[0]):
        for i, model in enumerate(Models):
            if Mapping_Dict["Mapping_Matrix"][j, i] != model.Glc_Index:
                Models[i].reactions[Mapping_Dict["Mapping_Matrix"][j, i]
                                    ].upper_bound = General_Uptake_Kinetics(C[j+Models.__len__()-1])
            else:
                Models[i].reactions[Mapping_Dict["Mapping_Matrix"][j, i]
                                    ].upper_bound = Glucose_Uptake_Kinetics(C[j])
            Models[i].reactions[Params["Amylase_Ind"]
                                [i]].lower_bound = (lambda x, a: a*x)(Models[i].Policy[(C(Params["Glucose_Index"]), C(Params["Starch_Index"]))], 5)

    for i in range(Models.__len__()):
        Models[i].optimize()
        dCdt[i] += Models[i].objective_value*C[i]

    for i in range(Mapping_Dict["Mapping_Matrix"].shape[0]):
        for j in range(Models.__len__()):
            if Mapping_Dict["Mapping_Matrix"][i, j] != -1:
                dCdt[i+Models.__len__()] += Models[j].reactions[Mapping_Dict["Mapping_Matrix"]
                                      [i, j]].x*C[i+Models.__len__()]

    dCdt[Models.__len__()+Mapping_Dict["Mapping_Matrix"].shape[0]] = Starch_Degradation_Kinetics(
    return dCdt


def Find_Optimal_Policy(Models, Mapping_Dict, ICs, Params):
    pass


def Generate_Episode():
    pass


def Build_Mapping_Matrix(Models):
    """
    Given a list of COBRA model objects, this function will build a mapping matrix

    """

    Ex_sp=[]
    for model in Models:
        for Ex_rxn in model.exchanges:
            if Ex_rxn.id not in Ex_sp:
                Ex_sp.append(Ex_rxn.id)
    Mapping_Matrix=np.zeros((len(Ex_sp), len(Models)), dtype=int)
    for i, id in enumerate(Ex_sp):
        for j, model in enumerate(Models):
            if id in model.reactions:
                Mapping_Matrix[i, j]=model.reactions.index(id)
            else:
                Mapping_Matrix[i, j]=-1
    return {"Ex_sp": Ex_sp, "Mapping_Matrix": Mapping_Matrix, "Models": Models}


def Build_ODE_System(Models, Mapping_Dict):
    pass


class Policy:
    """
    Any policy would be an instance of this class
    Given a state it will retrun a dictionary including the probability distribution o
    each action
    """

    def __init__(self, Porb_Dist_Dict):
        self.Policy=Porb_Dist_Dict

    def get_action(self, state):
        Actions=[(action, self.Policy[state][action])
                   for action in self.Policy[state].keys()]
        np.random.choice(Actions, p=[action[1] for action in Actions], k=1)


def Starch_Degradation_Kinetics(a_Amylase: float, Starch: float, Model=""):
    """
    This function calculates the rate of degradation of starch
    a_Amylase Unit: mmol
    Starch Unit: mg

    """
    if Model == "Ecoli":
        return a_Amylase*Starch


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


if __name__ == "__main__":
    main()

 # SCRATCH PAPER
#  Solution=scipy.integrate.solve_ivp(ODE_Sys, (0, 10),  ICs, t_eval=np.linspace(
#     0, 10, num=1000), method='Radau', args=[Params])
