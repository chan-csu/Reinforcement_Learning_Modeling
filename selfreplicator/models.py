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

    def __call__(self,x:float)->float:
        return self.kcat*x/(self.k + x)

class PingPong(Kinetic):
    """Objects of this class represent a Ping Pong kinetic model"""
    def __init__(self, parameters:dict)->None:
        if {"ka", "kb","kab","vm"} != set(parameters.keys()):
            raise ValueError("Ping Pong kinetic model requires only parameters ka and kb and kab and vm")
        super().__init__("PingPong", parameters)

    def __call__(self,a:float,b:float)->float:
        return self.vm*a*b/(self.ka*a + self.kb*b + self.ka*self.kb)


class Shape:
    """Objects of this class represent the shape of a cell"""
    def __init__(self, name:str,dimensions:tuple[dict[str,float]]):
        self.name = name
        self.dimensions = dimensions
        for key, value in dimensions.items():
            setattr(self, key, value)
    
    def __str__(self) -> str:
        dim_print=""
        for key, value in self.__dict__.items():
            dim_print += f"{key} = {value}\n"
        return f"{self.name} shape with dimensions:\n{dim_print}"
    
    def __repr__(self) -> str:
        return f"Shape({self.name},{self.dimensions})"
    
    @property
    def volume(self)->float:
        pass
    
    @property
    def area(self)->float:
        pass
    
    
    
class Sphere(Shape):
    """Objects of this class represent a sphere. 
    NOTE: This is hollow sphere. So, the constructor takes in a dictionary with the following
    keys: r,t. r is the radius of the sphere and t is the thickness of the sphere.
    The assumption in this class is that the thickness is uniform and throughout the sphere.
    """
    def __init__(self, dimensions:dict[str,float])->None:
        if {"r,t"} != set(dimensions.keys()):
            raise ValueError("Sphere shape requires only parameter r")
        super().__init__("Sphere", dimensions)
    
    @property
    def volume(self)->float:
        return 4/3*np.pi*self.r**3
    
    @property
    def area(self)->float:
        return 4*np.pi*self.r**2
    
    def calculate_differentials(self,dv:float)->dict[str,float]:
        """V=4*pi*[3rt^3+3t^2r+t^3] this equation should be used:
        dV=4*pi*[6rt+3t^2]*dr
        dr=dv/(4*pi*[6rt+3t^2])
        """
        diffs={k:0 for k in self.dimensions.keys()}
        diffs.update({"r":dv/(4*np.pi*(6*self.r*self.t + 3*self.t**2))})
        return diffs
    

    
        
def toy_model_stoichiometry(x:np.ndarray,params:dict)->np.ndarray:
    """This function returns the stoichiometry of the toy model. Here is a look at the governing rules:
    r[1]: S -> S_in                     ::: r=PingPong(ka1, kb1, kab1, vm1)([S]   ,[t1])
    r[2]: S_in -> p21 I_1 + p22 E       ::: r=PingPong(ka2, kb2, kab2, vm2)([S_in],[e2])
    r[3]: I_1 -> p31 P + p32 E          ::: r=PingPong(ka3, kb3, kab3, vm3)([I_1] ,[e3])
    r[4]: S_in + r41 E ->  p42 NTP      ::: r=PingPong(ka4, kb4, kab4, vm4)([S_in],[e4])
    r[5]: NTP -> p51 NA                 ::: r=PingPong(ka5, kb5, kab5, vm5)([NTP] ,[e5])
    r[6]: I_1 + r61 E -> p62 Li         ::: r=PingPong(ka6, kb6, kab6, vm6)([I_1] ,[e6])
    r[7]: I_1 + r71 E -> p72 AA         ::: r=PingPong(ka7, kb7, kab7, vm7)([I_1] ,[e7])
    r[8]: AA + r81 E -> p82 e           ::: r=PingPong(ka8, kb8, kab8, vm8)([AA]  ,[e8])
    r[9]: P->P_out                      ::: r=Hill(n1,k1)([t2])
    r[10]: r101 AA + r102 Li -> W       ::: r=PingPong(ka9, kb9, kab9, vm9)([AA],[Li]) -> Should be very fast    
    _________________________________________________________________________________________________________
    e=e1+e2+e3+e4+e5+e6+e7+e8+t1       
    #Q: is this okay?

    r[]:e->t1
    r[]:e->e1
    r[]:e->e2
    r[]:e->e3
    r[]:e->e4
    r[]:e->e5
    r[]:e->e6
    r[]:e->e7
    r[]:e->e8
    r[]:e->t2
    ____________________________________________________________________________________________
    Two quantities to track:
    Volume:
    Area:
    

    
    
    """
    