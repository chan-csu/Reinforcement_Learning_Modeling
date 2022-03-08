
# Script for running Community Dynamic Flux Balance Analysis (CDFBA)
# Written by: Parsa Ghadermazi

import numpy as np
import cobra
import os
Main_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    Number_of_Models = 2
    Models=[]
    Main_dir = os.path.dirname(os.path.abspath(__file__))
    Base_Model = cobra.io.read_sbml_model(Main_dir+'/IJO1366_AP.xml')
    [Models.append(Base_Model.copy()) for i in range(Number_of_Models)]
    for i in range(Number_of_Models):
        Models[i].name = "Ecoli_"+str(i+1)
    
    dFBA(Models, Init_C, Inlet_C)

def dFBA(Models, Init_C, Inlet_C):
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
    """
def Build_Mapping_Matrix(Models):
    Ex_sp=[]
    [Ex_sp.append(Model.name) for Model in Models]
    for model in Models:
        for Ex_rxn in model.exchanges:
            if Ex_rxn.id not in Ex_sp:
                Ex_sp.append(Ex_rxn.id)
    for model in Models:
        


    



class Policy:
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


if __name__ == "__main__":
    main()
