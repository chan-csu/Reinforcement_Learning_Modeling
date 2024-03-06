"""This module contains a variety of kinetic models to be used"""
import numpy as np
import scipy.integrate as integrate
import plotly.express as px
import pandas as pd

TOY_REACTIONS = [
    "S_import",
    "S_to_I1",
    "I1_to_P",
    "S_to_NTP",
    "NTP_to_NA",
    "I1_to_Li",
    "I1_to_AA",
    "AA_to_e",
    "P_export",
    "AA_and_li_to_W",
    "e_to_t1",
    "e_to_e1",
    "e_to_e2",
    "e_to_e3",
    "e_to_e4",
    "e_to_e5",
    "e_to_e6",
    "e_to_e7",
    "e_to_e8",
    "e_to_t2",
    ]

TOY_SPECIES = [
    "S_env",
    "S",
    "I1",
    "P",
    "NTP",
    "NA",
    "Li",
    "AA",
    "e",
    "t1",
    "t2",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
    "e6",
    "e7",
    "e8",
    "E",
    "W"
]
    
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
    
    def calculate_differentials(self,dv:float)->dict[str,float]:
        pass
    

class Sphere(Shape):
    """Objects of this class represent a sphere. 
    NOTE: This is hollow sphere. So, the constructor takes in a dictionary with the following
    keys: r,t. r is the radius of the sphere and t is the thickness of the sphere.
    The assumption in this class is that the thickness is uniform and throughout the sphere.
    """
    def __init__(self, dimensions:dict[str,float])->None:
        if {"r","t"} != set(dimensions.keys()):
            raise ValueError("Sphere shape requires only parameter r")
        super().__init__("Sphere", dimensions)
    
    @property
    def volume(self)->float:
        return 4/3*np.pi*self.r**3
    
    @property
    def area(self)->float:
        return 4*np.pi*self.r**2
    
    def set_dimensions(self,dimensions:dict[str,float])->None:
        self.dimensions = dimensions
        for key, value in dimensions.items():
            setattr(self, key, value)
    
    def calculate_differentials(self,dv:float)->dict[str,float]:
        """V=4*pi*[3rt^3+3t^2r+t^3] this equation should be used:
        dV=4*pi*[6rt+3t^2]*dr
        dr=dv/(4*pi*[6rt+3t^2])
        """
        diffs={k:0 for k in self.dimensions.keys()}
        diffs.update({"r":dv/(4*np.pi*(6*self.r*self.t + 3*self.t**2))})
        return diffs
    





class Cell:
    """Objects of this class represent the biological function of a Cell"""
    def __init__(self,
                 name:str,
                 stoichiometry:callable,
                 ode_sys:callable, 
                 parameters:dict,
                 reactions:list,
                 compounds:list,
                 shape:Shape):
        self.name = name
        self.stoichiometry = stoichiometry
        self.ode_sys = ode_sys
        self.parameters = parameters
        self.shape=shape
        self.reactions = reactions
        self.compounds = compounds
        self.state_variables = self.get_state_variables()
        self.kinetics={}
    
    def get_state_variables(self)->list:
        """
        This method returns the state variables of the cell. The state variables are the compounds and the shape variables\
        
        """
        state_variables = self.compounds.copy()
        shape_variables = [key for key in self.shape.dimensions.keys()]
        state_variables.extend(shape_variables)
        state_variables.append("Volume")
        return state_variables
class Environment:
    pass
    
        

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
    
class Linear(Kinetic):
    """Objects of this class represent a Linear kinetic model"""
    def __init__(self, parameters:dict)->None:
        if {"k"} != set(parameters.keys()):
            raise ValueError("Linear kinetic model requires only parameters k")
        super().__init__("Linear", parameters)

    def __call__(self,x:float)->float:
        return self.k*x


def toy_model_stoichiometry(model:Cell)->np.ndarray:
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
    r[10]: r101 AA + r102 Li -> W       ::: r=PingPong(ka10, kb10, kab10, vm10)([AA],[Li]) -> Should be very fast    
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
    
    """
    s=np.zeros((len(model.state_variables),len(model.reactions)))
    s[[model.state_variables.index("S_env"),model.state_variables.index("S")],model.reactions.index("S_import")] = [-1,1]
    s[list(map(model.state_variables.index,["S",
    "I1", "E"])),model.reactions.index("S_to_I1")] = [-1,model.parameters["p21"],model.parameters["p22"]]
    
    s[list(map(model.state_variables.index,["I1",
    "P",
    "E"])),model.reactions.index("I1_to_P")] = [-1,model.parameters["p31"],model.parameters["p32"]]
    s[list(map(model.state_variables.index,["S","E",
    "NTP"])),model.reactions.index("S_to_NTP")] = [-1,model.parameters["r41"],model.parameters["p42"]]
    s[list(map(model.state_variables.index,["NTP","NA"])),model.reactions.index("NTP_to_NA")] = [-1,model.parameters["p51"]]
    s[list(map(model.state_variables.index,["I1",
    "E",
    "Li"])),model.reactions.index("I1_to_Li")] = [-1,model.parameters["r61"],model.parameters["p62"]]
    s[list(map(model.state_variables.index,["I1","E","AA"])),
      model.reactions.index("I1_to_AA")] = [-1,model.parameters["r71"],model.parameters["p72"]]
    
    s[list(map(model.state_variables.index,["AA","E","e"])),
      model.reactions.index("AA_to_e")] = [-1,model.parameters["r81"],model.parameters["p82"]]
    
    s[[model.state_variables.index("P")],model.reactions.index("P_export")] = -1
    
    s[list(map(model.state_variables.index,["AA","Li","W"])),
      model.reactions.index("AA_and_li_to_W")] = [model.parameters["r101"],model.parameters["r102"],1]
    
    s[list(map(model.state_variables.index,["e","t1"])),model.reactions.index("e_to_t1")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e1"])),model.reactions.index("e_to_e1")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e2"])),model.reactions.index("e_to_e2")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e3"])),model.reactions.index("e_to_e3")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e4"])),model.reactions.index("e_to_e4")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e5"])),model.reactions.index("e_to_e5")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e6"])),model.reactions.index("e_to_e6")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e7"])),model.reactions.index("e_to_e7")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e8"])),model.reactions.index("e_to_e8")] = [-1,1]
    s[list(map(model.state_variables.index,["e","t2"])),model.reactions.index("e_to_t2")] = [-1,1]
    return s

    
def toy_model_ode(t, y, model:Cell)->np.ndarray:
    ### First we update the dimensions of the cell
    model.shape.set_dimensions({key:y[model.state_variables.index(key)] for key in model.shape.dimensions.keys()})
    y[model.state_variables.index("Volume")]=model.shape.volume
    ### Now we calculate the fluxes for each reaction
    fluxes = np.zeros(len(model.reactions))
    fluxes[model.reactions.index("S_import")] = model.kinetics.setdefault("S_import",
                                                                          PingPong({"ka": model.parameters["ka1"],
                                                                                    "kb": model.parameters["kb1"],
                                                                                    "kab":model.parameters["kab1"],
                                                                                    "vm": model.parameters["vm1"]}))\
                                                                            (y[model.state_variables.index("S_env")],
                                                                             y[model.state_variables.index("t1")]/model.shape.area)*model.shape.area
    
    fluxes[model.reactions.index("S_to_I1")] = model.kinetics.setdefault("S_to_I1",
                                                                        PingPong({"ka":  model.parameters["ka2"],
                                                                                  "kb":  model.parameters["kb2"],
                                                                                  "kab": model.parameters["kab2"],
                                                                                  "vm":  model.parameters["vm2"]}))\
                                                                            (y[model.state_variables.index("S")]/model.shape.volume,
                                                                             y[model.state_variables.index("e2")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("I1_to_P")] = model.kinetics.setdefault("I1_to_P",
                                                                          PingPong({"ka": model.parameters["ka3"],
                                                                                    "kb": model.parameters["kb3"],
                                                                                    "kab":model.parameters["kab3"],
                                                                                    "vm": model.parameters["vm3"]}))\
                                                                             (y[model.state_variables.index("I1")]/model.shape.volume,
                                                                              y[model.state_variables.index("e3")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("S_to_NTP")] = model.kinetics.setdefault("S_to_NTP",
                                                                            PingPong({"ka":  model.parameters["ka4"], 
                                                                                     "kb":  model.parameters["kb4"], 
                                                                                     "kab": model.parameters["kab4"], 
                                                                                     "vm":  model.parameters["vm4"]}))\
                                                                                (y[model.state_variables.index("S")]/model.shape.volume,
                                                                                y[model.state_variables.index("e4")]/model.shape.volume)*model.shape.volume*y[model.state_variables.index("E")]
    
    fluxes[model.reactions.index("NTP_to_NA")] = model.kinetics.setdefault("NTP_to_NA",
                                                                            PingPong({"ka":  model.parameters["ka5"],
                                                                                      "kb":   model.parameters["kb5"],
                                                                                      "kab":  model.parameters["kab5"],
                                                                                      "vm":   model.parameters["vm5"]}))\
                                                                                (y[model.state_variables.index("NTP")]/model.shape.volume,
                                                                                 y[model.state_variables.index("e5")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("I1_to_Li")] = model.kinetics.setdefault("I1_to_Li",
                                                                            PingPong({"ka":  model.parameters["ka6"],
                                                                                      "kb":  model.parameters["kb6"],
                                                                                      "kab": model.parameters["kab6"],
                                                                                      "vm":  model.parameters["vm6"]}))\
                                                                                (y[model.state_variables.index("I1")]/model.shape.volume,
                                                                                 y[model.state_variables.index("e6")]/model.shape.volume)*model.shape.volume*y[model.state_variables.index("E")]
    
    fluxes[model.reactions.index("I1_to_AA")] = model.kinetics.setdefault("I1_to_AA",
                                                                            PingPong({  "ka": model.parameters["ka7"],
                                                                                        "kb": model.parameters["kb7"],
                                                                                        "kab":model.parameters["kab7"],
                                                                                        "vm": model.parameters["vm7"]}))\
                                                                                (y[model.state_variables.index("I1")]/model.shape.volume,
                                                                                 y[model.state_variables.index("e7")]/model.shape.volume)*model.shape.volume*y[model.state_variables.index("E")]
    
    fluxes[model.reactions.index("AA_to_e")] = model.kinetics.setdefault("AA_to_e",
                                                                        PingPong({"ka":   model.parameters["ka8"],
                                                                                  "kb":   model.parameters["kb8"],
                                                                                  "kab":  model.parameters["kab8"],
                                                                                  "vm":   model.parameters["vm8"]}))\
                                                                            (y[model.state_variables.index("AA")]/model.shape.volume,
                                                                             y[model.state_variables.index("e8")]/model.shape.volume)*model.shape.volume*y[model.state_variables.index("E")]
                                                                            
    fluxes[model.reactions.index("P_export")] = model.kinetics.setdefault("P_export",
                                                                        PingPong({
                                                                            "ka": model.parameters["ka9"],
                                                                            "kb": model.parameters["kb9"],
                                                                            "kab":model.parameters["kab9"],
                                                                            "vm": model.parameters["vm9"]}))(y[model.state_variables.index("P")]/model.shape.volume,
                                                                            y[model.state_variables.index("t2")]/model.shape.area)*model.shape.area
                                                                            
    fluxes[model.reactions.index("AA_and_li_to_W")] = model.kinetics.setdefault("AA_and_li_to_W",
                                                                                 PingPong({ "ka": model.parameters["ka10"],
                                                                                           "kb": model.parameters["kb10"],
                                                                                           "kab":model.parameters["kab10"],
                                                                                           "vm": model.parameters["vm10"]}))\
                                                                                  (y[model.state_variables.index("AA")]/model.shape.volume,
                                                                                    y[model.state_variables.index("Li")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_t1")] = model.kinetics.setdefault("e_to_t1",
                                                                        Linear({"k":model.parameters["k_t1"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e1")] = model.kinetics.setdefault("e_to_e1",
                                                                        Linear({"k":model.parameters["k_e1"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume

    fluxes[model.reactions.index("e_to_e2")] = model.kinetics.setdefault("e_to_e2",
                                                                        Linear({"k":model.parameters["k_e2"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e3")] = model.kinetics.setdefault("e_to_e3",
                                                                        Linear({"k":model.parameters["k_e3"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e4")] = model.kinetics.setdefault("e_to_e4",
                                                                        Linear({"k":model.parameters["k_e4"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume

    fluxes[model.reactions.index("e_to_e5")] = model.kinetics.setdefault("e_to_e5",
                                                                        Linear({"k":model.parameters["k_e5"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e6")] = model.kinetics.setdefault("e_to_e6",
                                                                        Linear({"k":model.parameters["k_e6"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e7")] = model.kinetics.setdefault("e_to_e7",
                                                                        Linear({"k":model.parameters["k_e7"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e8")] = model.kinetics.setdefault("e_to_e8",
                                                                        Linear({"k":model.parameters["k_e8"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_t2")] = model.kinetics.setdefault("e_to_t2",
                                                                        Linear({"k":model.parameters["k_t2"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume

                                                                         
    v=np.matmul(model.stoichiometry(model),fluxes) 
    dvdt=model.parameters["lipid_density"]*v[model.state_variables.index("W")]
    for dim in model.shape.dimensions.keys():
        v[model.state_variables.index(dim)] = model.shape.calculate_differentials(dvdt)[dim]
    
    return v

if __name__ == "__main__":
    s=Sphere({"r":0.1,"t":0.5})
    cell=Cell("Toy Model",
              toy_model_stoichiometry,
              toy_model_ode,
              {"ka1":100,
               "kb1":10,
               "kab1":1,
               "vm1":100,
               "ka2":1,
               "kb2":1,
               "kab2":1,
               "vm2":1,
               "ka3":1,
               "kb3":1,
               "kab3":1,
               "vm3":1,
               "ka4":1,
               "kb4":1,
               "kab4":1,
               "vm4":1,
               "ka5":1,
               "kb5":1,
               "kab5":1,
               "vm5":1,
               "ka6":1,
               "kb6":1,
               "kab6":1,
               "vm6":0.001,
               "ka7":1,
               "kb7":1,
               "kab7":1,
               "vm7":1,
               "ka8":1,
               "kb8":1,
               "kab8":1,
               "vm8":1,
               "ka9":1,
               "kb9":1,
               "kab9":1,
               "vm9":1,
               "ka10":1,
               "kb10":1,
               "kab10":1,
               "vm10":1,
               "k_t1":10,
               "k_e1":1,
               "k_e2":1,
               "k_e3":1,
               "k_e4":1,
               "k_e5":1,
               "k_e6":1,
               "k_e7":1,
               "k_e8":1,
               "k_t2":1,
               "p21":1,
               "p22":1,
               "p31":1,
               "p32":1,
               "r41":-1,
               "p42":1,
               "p51":1,
               "r61":-1,
               "p62":1,
               "r71":-1,
               "p72":1,
               "r81":-1,
               "p82":1,
               "r101":-1,
                "r102":-1,
               "lipid_density":0.001},
              TOY_REACTIONS,
              TOY_SPECIES,
              s)
    c0=np.ones(len(cell.state_variables))/100
    c0[cell.state_variables.index("S_env")]=10
    c0[cell.state_variables.index("r")]=s.r
    c0[cell.state_variables.index("Volume")]=s.volume
    sol=integrate.solve_ivp(toy_model_ode,(0,1000),c0,args=(cell,),method="RK23",t_eval=np.linspace(0,1000,1000))
    sol_df=pd.DataFrame(sol.y,index=cell.state_variables,columns=sol.t)
    sol_df=sol_df/sol_df.loc["Volume"]
    px.line(sol_df.T).show()

        
    