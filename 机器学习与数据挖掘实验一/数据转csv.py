import pandas as pd
import random
df1=pd.read_excel('数据库随机表.xlsx',encoding='gbk')
df2=pd.read_excel('数据库随机表.xlsx',encoding='gbk')
row_list=[random.randint(0,99) for x in range(70) ]
df3=df1.drop(row_list,axis=0)
df4=df1.drop(df3.index)
# print(df3)
# print(df4)
df5=df4.sample(frac=0.5)
# print(df5)
result=pd.concat([df3,df5])
result.to_csv('数据库表1.csv')
Gender=df4['Gender'].tolist()
Gender_new=[]
for i in Gender:
    if i=='girl':
        Gender_new.append('female')
    if i=='boy':
        Gender_new.append('male')
df4['Gender']=Gender_new
df4['Height']=df4['Height'].apply(lambda x:x/100)
df4.to_csv('txt表1.csv')
print(result)

