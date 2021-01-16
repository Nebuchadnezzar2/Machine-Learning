import pandas as pd

df=pd.read_csv('合并后的数据.csv',encoding='gbk')
# print(df)
df1=df[(df['City'])=='Beijing']
df2=df[(df['City']=='Guangzhou')&(df['C1']>=80)&(df['C10']>=9)&(df['Gender']=='boy')].shape[0]

# print(df1)
C1_mean=df1['C1'].mean()
C2_mean=df1['C2'].mean()
C3_mean=df1['C3'].mean()
C4_mean=df1['C4'].mean()
C5_mean=df1['C5'].mean()
C6_mean=df1['C6'].mean()
C7_mean=df1['C7'].mean()
C8_mean=df1['C8'].mean()
C9_mean=df1['C9'].mean()
C10_mean=df1['C10'].mean()
print('北京学生C1课程平均成绩 % .2f'% C1_mean)
print('北京学生C2课程平均成绩 %.2f'% C2_mean)
print('北京学生C3课程平均成绩 % .2f'% C3_mean)
print('北京学生C4课程平均成绩 % .2f'% C4_mean)
print('北京学生C5课程平均成绩 % .2f'% C5_mean)
print('北京学生C6课程平均成绩 % .2f'% C6_mean)
print('北京学生C7课程平均成绩 % .2f'% C7_mean)
print('北京学生C8课程平均成绩 % .2f'% C8_mean)
print('北京学生C9课程平均成绩 % .2f'% C9_mean)
print('北京学生C10课程平均成绩 % .2f'% C10_mean)
print('广州男学生C1成绩80分以上并且C10成绩9分以上数量为%.2f'%df2)

