import numpy as np
import dearpygui.dearpygui as dpg

class Display:
    
    def __init__(self,
                 state:np.ndarray,
                 config:dict):
        self.state = state
        self.config = config
    
    def render(self):
        pass
    
    def get_state(self):
        pass

def render_in_dear_pygui(state:np.ndarray,config:dict):
    dpg.create_context()
    dpg.create_viewport(title=config["window_title"], 
                        width=config["viewport_w"], 
                        height=config["viewport_h"])

    dpg.destroy_context()