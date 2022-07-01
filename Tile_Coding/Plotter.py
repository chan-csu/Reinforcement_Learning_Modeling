from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd

def main(t):
    Data=pd.read_table("F.csv",delimiter=",")
    Agents=Data.columns[1:]
    plt.cla()
    for i in Agents:
        plt.plot(Data[i])
    plt.legend(Agents)


t=FuncAnimation(plt.gcf(),main,interval=1000)
plt.show()