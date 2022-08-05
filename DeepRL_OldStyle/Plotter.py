from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
_,ax=plt.subplots(2,1)
def main(t):
    Data=pd.read_table("F.csv",delimiter=",")
    Cols=Data.columns[1:]
    ax[0].cla()
    ax[1].cla()
    for i in range(Cols.__len__()):
        if i%2==0:
            ax[0].plot(Data[Cols[i]])
        else:
            ax[1].plot(Data[Cols[i]])





t=FuncAnimation(plt.gcf(),main,interval=1000)
plt.show()