import pandas as pd
import matplotlib.pyplot as plt
from 实验三四问 import cons2num
import numpy as np
from matplotlib.pyplot import MultipleLocator

df=pd.read_csv('合并数据.csv',encoding='gbk')
# print(df.head())
C1_arr=df['C1'].tolist()
Cons_arr=df['Constitution'].apply(cons2num).tolist()
Cons_xticks=np.arange(0,105,5)
# print(C1_arr)
# print(Cons_arr)
# print(Cons_xticks)
fig=plt.figure(figsize=(20,10),num='C1与体能成绩散点图')
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
ax2_major_locator = MultipleLocator(5)
ax1.scatter(C1_arr,Cons_arr,color='red',s=30)
# ax2.set_xlim([0,100])
max_value = max(C1_arr)
min_value = min(C1_arr)
num_bin = int((max_value-min_value)//5)
ax2.set_xlim([0,100])
ax2.xaxis.set_major_locator(ax2_major_locator)
ax2.hist(C1_arr,bins=num_bin)
# ax2.set_xlim([0,100])
# ax2.xaxis.set_major_locator(ax2_major_locator)

plt.show()
df['Constitution']=df['Constitution'].apply(cons2num)
def z_score(z_score_list,df):
    for i in z_score_list:
        mean_value=df[i].mean()
        s_value=0
        print(mean_value)
        for j in df[i]:
            s_value+=(j-mean_value)**2
        print(s_value)
        s_value=pow((s_value/df[i].count()),0.5)
        print(s_value)
        df[i]=df[i].apply(lambda x:(x-mean_value)/s_value)
        print(df[i])
z_score_list=['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','Constitution']
z_score(z_score_list,df)
# df.to_csv('归一化的数据.csv')

