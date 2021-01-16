import pandas as pd
df=pd.read_csv('合并后的数据.csv',encoding='gbk')
df1=df[(df['City']=='Guangzhou')&(df['Gender']=='girl')]
df2=df[(df['City']=='Shanghai')&(df['Gender']=='girl')]
# print(df1)
# print(df2)
def cons2num(x):
    if x=='excellent':
        return 90
    elif x=='good':
        return 85
    elif x=='general':
        return 70
    elif x=='bad':
        return 60
GZ_PE_mean=df1['Constitution'].apply(cons2num).mean()
SH_PE_mean=df2['Constitution'].apply(cons2num).mean()
if GZ_PE_mean>SH_PE_mean:
    print('广州女生平均体能测试成绩比上海女生好\r\n平均体能测试成绩分别为%.2f、%.2f\r\n'% (GZ_PE_mean,SH_PE_mean))
elif GZ_PE_mean==SH_PE_mean:
    print('广州女生平均体能成绩与上海女生实力相当\r\n平均体能测试成绩分别为%.2f、%.2f\r\n'% (GZ_PE_mean,SH_PE_mean))
else:
    print('广州女生平均体能成绩比上海女生差\r\n平均体能测试成绩分别为%.2f、%.2f\r\n'% (GZ_PE_mean,SH_PE_mean))




# print(GZ_PE_mean)
# print(SH_PE_mean)
# print(df1)
col_name1=['C1','C2','C3','C4','C5']
col_name2=['C6','C7','C8','C9','C10']
mul_10_grade=df[col_name2].apply(lambda x:x*10)
academic_mean=(df[col_name1].sum().sum()+mul_10_grade.sum().sum())/1000
pe_mean=df['Constitution'].apply(cons2num).mean()
each_aca_mean=df[col_name1].apply(lambda x:x.sum(),axis=1)+mul_10_grade.apply(lambda x:x.sum(),axis=1)
each_aca_mean=each_aca_mean.apply(lambda x:x-academic_mean)
each_pe_mean=df['Constitution'].apply(cons2num)
each_pe_mean=each_pe_mean.apply(lambda x:x-pe_mean)
cov=(each_aca_mean*each_pe_mean/104).sum()
std_aca=pow(pow(each_aca_mean,2).sum()/104,0.5)
std_pr=pow(pow(each_pe_mean,2).sum()/104,0.5)
std_aca_pe=pow(std_aca*std_pr,0.5)
correlation=cov/std_aca_pe
# print(each_aca_mean)
# print(each_pe_mean)
# print(pe_mean)
print("学习成绩和体能测试成绩的相关系数：%s" % correlation)
if correlation<0.3:
    print("得出结论：学习成绩与体能成绩不相关")
elif correlation>=0.3 and correlation<=0.8:
    print("得出结论：学习成绩与体能成绩弱相关")
else:
    print("得出结论：学习成绩与体能成绩相关")

