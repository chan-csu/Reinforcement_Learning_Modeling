"""This module contains a variety of kinetic models to be used"""

import numpy as np
import scipy.integrate as integrate

class Cell:
    """Objects of this class represent the biological function of a Cell"""
    def __init__(self, name:str, stoichiometry:callable, parameters:dict):
        self.name = name
        self.stoichiometry = stoichiometry
        self.parameters = parameters

class Kinetic:
    """Objects of this class represent a kinetic model"""
    def __init__(self, name:str, parameters:dict)->None:
        self.name = name
        for key, value in parameters.items():
            setattr(self, key, value)
    
    def __str__(self) -> str:
        param_print=""
        for key, value in self.__dict__.items():
            param_print += f"{key} = {value}\n"
        return f"{self.name} kinetic model with parameters:\n{param_print}"
        
class Hill(Kinetic):
    """Objects of this class represent a Hill kinetic model"""
    def __init__(self,parameters:dict)->None:
        if {"n", "k"} != set(parameters.keys()):
            raise ValueError("Hill kinetic model requires only parameters n and k")
        super().__init__("Hill", parameters)

    def __call__(self,x)->float:
        return x**self.n/(self.k**self.n + x**self.n)

class MichaelisMenten(Kinetic):
    """Objects of this class represent a Michaelis Menten kinetic model"""
    def __init__(self, parameters:dict)->None:
        if {"k", "kcat"} != set(parameters.keys()):
            raise ValueError("Michaelis Menten kinetic model requires only parameters k and kcat")
        super().__init__("MichaekisMenten", parameters)

    def __call__(self,x)->float:
        return self.kcat*x/(self.k + x)

class PingPong(Kinetic):
    """Objects of this class represent a Ping Pong kinetic model"""
    def __init__(self, parameters:dict)->None:
        if {"ka", "kb","vm"} != set(parameters.keys()):
            raise ValueError("Ping Pong kinetic model requires only parameters k1 and k2")
        super().__init__("PingPong", parameters)

    def __call__(self,a:float,b:float)->float:
        return self.vm*a*b/(self.ka*a + self.kb*b + self.ka*self.kb)

        
def toy_model_stoichiometry(x:np.ndarray,params:dict)->np.ndarray:
    """This function returns the stoichiometry of the toy model. Here is a look at the biochemical reaction:
    
    """
    