import pickle as pkl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 

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
        

class CompareDataProcessor:
    def __init__(self,
                 data:dict[str:BatchDataProcessor],
                ):
        self.data = data
        
    def compare_states(self,agent:str,on:str='all')->None:
        collector={key:pd.DataFrame(pd.DataFrame(value._data[agent]["s"].numpy().reshape((-1,value.batch_size,len(value.state_vars[agent])),order='F').mean(axis=1),columns=value.state_vars[agent])) for key,value in self.data.items()}
        if on=='all':
            fig=go.Figure()
            for key in collector.keys():
                fig.add_trace(go.Scatter(x=collector[key].index,y=collector[key].mean(axis=1),mode='lines',name=key))
            
            fig.show()
        else:
            fig=go.Figure()
            for key in collector.keys():
                fig.add_trace(go.Scatter(x=collector[key].index,y=collector[key][on],mode='lines',name=key))

            fig.show()
    
        
if __name__ == "__main__":
    data1 = "/Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/selfreplicator/data_batch_0.pkl"
    data2 = "/Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/selfreplicator/data_batch_200.pkl"
    data3 = "/Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/selfreplicator/data_batch_400.pkl"
    data4 = "/Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/selfreplicator/data_batch_600.pkl"
    data5 = "/Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/selfreplicator/data_batch_800.pkl"
    data6 = "/Users/parsaghadermarzi/Desktop/Academics/Projects/Reinforcement_Learning_Modeling/selfreplicator/data_batch_2000.pkl"
    compare=CompareDataProcessor(
        {
            "data1":BatchDataProcessor(data1,8),
            "data2":BatchDataProcessor(data2,8),
            "data3":BatchDataProcessor(data3,8),
            "data4":BatchDataProcessor(data4,8),
            "data5":BatchDataProcessor(data5,8),
            "data6":BatchDataProcessor(data6,8),
        }
    )
    compare.compare_states("Toy Model","P_env")
    
    
    


    
    
