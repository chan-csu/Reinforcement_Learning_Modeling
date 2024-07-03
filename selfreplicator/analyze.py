import pickle as pkl
import pandas as pd
import plotly.express as px

class BatchDataProcessor:
    def __init__(self,
                 data:str,
                 batch_size:int,
                 ):
        self.data = data
        self.batch_size = batch_size
        
        with open(self.data, 'rb') as f:
            self._data = pkl.load(f)
        self.agents=list(self._data.keys())
        self.state_vars = {}
        self.action_vars={}
        for agent in self._data.keys():
            self.state_vars[agent]=list(self._data[agent]["state_vars"])
            self.action_vars[agent]=list(self._data[agent]["action_vars"])
        
    def visualize_actions(self,agent:str):
        df=pd.DataFrame(self._data[agent]["a"].numpy().reshape((-1,self.batch_size,len(self.action_vars[agent])),order='F').mean(axis=1),columns=self.action_vars[agent])
        fig=px.line(df,x=df.index,y=df.columns)
        fig.show()
    
    def visualize_states(self,agent:str):

        df=pd.DataFrame(pd.DataFrame(self._data[agent]["s"].numpy().reshape((-1,self.batch_size,len(self.state_vars[agent])),order='F').mean(axis=1),columns=self.state_vars[agent]))
        fig=px.line(df,x=df.index,y=df.columns)
        fig.show()
        
        
if __name__ == "__main__":
    data = "/Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/selfreplicator/data_batch_0.pkl"
    batch_size = 8
    BatchDataProcessor(
        data=data,
        batch_size=batch_size,
    ).visualize_actions(agent="Toy Model")

    


    
    
