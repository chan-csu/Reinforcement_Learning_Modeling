
# Script for running Community Dynamic Flux Balance Analysis (CDFBA)
# Written by: Parsa Ghadermazi

import numpy as np
import cobra
import os
Main_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    pass


def dFBA(Models, Init_C, Inlet_C):
    """
    Main function for running Dynamic Flux Balance Analysis (dFBA)
    Models is a list of COBRA Model objects
    NOTE: this implementation of DFBA is compatible with RL framework
    Given a policy it will genrate episodes. Policies can be either deterministic or stochastic
    """


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
