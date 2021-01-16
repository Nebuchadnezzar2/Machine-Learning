import pandas as pd
import random
import csv

# # attention! 设置newline，否则会出现两行之间有一行空行
# with open('txt表1.csv', 'w', newline='') as csvfile: # 写入的目标csv文件路径
#     writer = csv.writer(csvfile)
#     data = open('txt表1.txt')  # 改成自己的路径
#     for each_line in data:
#         a = each_line.split()  # 我的数据每行都是字符串，所以将str按空格分割，变成list
#         writer.writerow(a)
# df=pd.read_csv('txt表1.txt')
# df.to_csv('txt表1.csv')

df1=pd.read_csv('txt表1.csv')
df1['C10']=df1['C10'].fillna(0)
Gender=df1['Gender'].tolist()
# print(df1)
Gender_new=[]
for i in Gender:
    if i=='female':
        Gender_new.append('girl')
    if i=='male':
        Gender_new.append('boy')
    if i=='boy':
        Gender_new.append('boy')
    if i=='girl':
        Gender_new.append('girl')

df1['Gender']=Gender_new
df1['Height']=df1['Height'].apply(lambda x:float(x*100))
df1['ID']=df1['ID'].apply(lambda x:str(x))

df2=pd.read_excel('数据库表1.xlsx')
df2['C10']=df2['C10'].fillna(0)
def f(x):
    if x<10:
        return '20200'+str(x)
    elif x>=100:
        return '202'+str(x)
    else:
        return '2020'+str(x)

df2['ID']=df2['ID'].apply(f)
df3=pd.concat([df1,df2],axis=0).drop_duplicates()
# df3=df3.sort_values('ID')
df3_dropna=df3.dropna(axis=0)
df3_unique=df3.drop_duplicates(subset=['ID'],keep=False)
# df3=df3.dropna(axis=0)
df3=pd.concat([df3_dropna,df3_unique],axis=0).drop_duplicates(subset=['ID']).sort_values('ID')
df3['Height']=df3['Height'].apply(lambda x:int(x/100) if x>1000 else x)
df3['Constitution']=df3['Constitution'].fillna('bad')



df3.to_csv('合并后的数据.csv')
# print(df3_unique)
# print(df3_dropna)
print(df3)

