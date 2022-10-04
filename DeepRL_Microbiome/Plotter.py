from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd

def main(t):
    Data=pd.read_table("Data.csv",delimiter=",",index_col=0)
    plt.clf()
    plt.plot(Data)
    plt.legend(Data.columns)
    plt.ylim(0,5)





t=FuncAnimation(plt.gcf(),main,interval=1000)
plt.show()