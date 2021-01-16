# Machine-Learning

实验一
====
实验一二问代码：
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

实验三四问代码：
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

注释
实验用Python完成，先进行了数据库的连接，在数据库中导入csv表后，
进行数据库表和txt表的合并，之后对合并的表进行清洗，对冗余和不一致性的进行并运算然后再删除重复，
填补缺失的数据(数值按列填补列浮点数平均值，体能成绩空值填补bad），
然后进行数据的规范化，对单位不同的数值进行一致规范，之后再进行相关的统计计算，求得对应问题的平均数，对体能测试的文字表述进行数值赋分再进行计算，
求出学习成绩与数值体能成绩的协方差与标准差，最后依照公式得出相关系数。

实验二
====
实验一二三代码：
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

实验四五代码：
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf) #矩阵全部显示
ori_df=pd.read_csv('归一化的数据.csv')
for col in ori_df.columns:
    ori_df[col]=ori_df[col].apply(lambda x:ori_df[col].mean() if x==0 else x)
ori_df['C10']=ori_df['Constitution']
zero_df=ori_df.iloc[:,7:17]
# zero_df.to_csv('归一化的成绩.csv')

print(zero_df.shape[0])
def r_matrix(df):
    matrix_r=np.empty(shape=(df.shape[0],df.shape[0]))
    for i in range(df.shape[0]):
        i_mean=df.iloc[i,:].mean() #求所求行的平均数
        for j in range(df.shape[0]):
            j_mean=df.iloc[j,:].mean()#求其余行平均数（包括所求行）
            X=df.iloc[i,:].apply(lambda x:(x-i_mean))
            Y=df.iloc[j,:].apply(lambda x:(x-j_mean))
            COVX_Y=(X*Y).sum(axis=0)
            VARX_Y=pow((pow(X,2).sum()*pow(Y,2).sum()),0.5)
            matrix_r[i][j]=round(COVX_Y/VARX_Y,3)
    return matrix_r

      # for col in col_list:
matrix_r=r_matrix(zero_df)
# np.savetxt('matrix_r.txt',matrix_r)  # 保存矩阵
print(matrix_r)
# print(matrix_r.shape)
def rel_ID(matrix_r):
    matrix=np.empty(shape=(matrix_r.shape[0],4))
    matrix=matrix.astype(np.str)
    matrix_r_index = np.argsort(-matrix_r,axis=1)
    for i in range(matrix_r_index.shape[0]):
        for j in range(4):
            index=matrix_r_index[i][j]
            matrix[i][j]=ori_df.loc[index,'ID']
    return matrix

rel_ID_matrix=rel_ID(matrix_r)
np.savetxt('rel_ID_matrix.txt',rel_ID_matrix,fmt='%s',delimiter='\t')
print(rel_ID_matrix)

注释
试验用Python完成，用包绘制出散点图与直方图
以课程1成绩为x轴，体能成绩为y轴，画出散点图。
以5分为间隔，画出课程1的成绩直方图。
对每门成绩进行z-score归一化，得到归一化的数据矩阵。
计算出100x100的相关矩阵，并可视化出混淆矩阵。（为避免歧义，这里“协相关矩阵”进一步细化更正为100x100的相关矩阵，100为学生样本数目，视实际情况而定）
根据相关矩阵，找到距离每个样本最近的三个样本，得到100x3的矩阵，其中第一列为测试，可忽略。

实验三
====
代码：
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def random_point(k):
    '''
    随机质心
    '''
    b = set()
    while(len(b) < k):
        b.add(np.random.randint(0, 150))
    return(b)
# 每个点到中心点距离距离


def get_distance(point_x, point_y, cent_x, cent_y, k):
    x = point_x
    y = point_y
    x0 = cent_x
    y0 = cent_y
    i = 0
    j = 0
    ds = [[]for i in range(len(x))]

    while i < len(x):
        while j < k:
            M = np.sqrt((x[i] - x0[j]) * (x[i] - x0[j]) +
                        (y[i] - y0[j]) * (y[i] - y0[j]))
            M = round(M, 1)
            j = j + 1
            ds[i].append(M)
        j = 0
        i = i + 1
    return(ds)


def error_distance(point_x, point_y, cent_x, cent_y, k):
    '''
    计算距离误差

    '''
    x = point_x
    y = point_y
    x0 = cent_x
    y0 = cent_y
    i = 0
    j = 0
    sum = 0
    while i < k:
        while j < len(x):
            M = (x[j] - x0[i]) * (x[j] - x0[i]) + \
                (y[j] - y0[i]) * (y[j] - y0[i])
            M = round(M, 1)
            sum += M
            j = j + 1
            # ds[i].append(M)
        j = 0
        i = i + 1
    return(sum)


def cent(lable):
    '''
    质心计算

    '''
    temp = lable
    mean_x = []
    mean_y = []
    i = 0
    j = 0
    while i < 3:
        cent_x = 0
        cent_y = 0
        count = 0
        while j < len(x):
            if i == temp[j]:
                count = count + 1
                cent_x = cent_x + x[j]
                cent_y = cent_y + y[j]
            j = j + 1
        cent_x = cent_x / count
        cent_y = cent_y / count
        # 更新中心点
        mean_x.append(cent_x)
        mean_y.append(cent_y)
        j = 0
        i = i + 1
    return[mean_x, mean_y]

# 按照k值聚类


def k_means(ds, x):
    x = x
    x = len(x)
    i = 0
    temp = []
    while i < x:
        temp.append(ds[i].index(min(ds[i])))
        i = i + 1
    return(temp)


def k_view():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(20, 10), num='聚类图与原图对比')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(x0, y0, color='r', s=50, marker='s')
    ax1.scatter(x, y, c=temp, s=25, marker='o')
    plt.xlabel('花萼面积Sepalarea')
    plt.ylabel('花瓣面积Petalarea')
    plt.title("聚类后的数据")
    ax2 = fig.add_subplot(1, 2, 2)
    plt.xlabel('花萼面积Sepalarea')
    plt.ylabel('花瓣面积Petalarea')
    plt.title("原来的数据")
    ax2.scatter(x, y, c=lable_code, s=25, marker='o')
    plt.show()


if __name__ == '__main__':
    iris_data = pd.read_csv('iris.csv')
    X = np.array(iris_data.iloc[:, 0:4])  # 特征向量，并且是按顺序排列的
    lable = np.array(iris_data.iloc[:, 4])  # 标签
    lable_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    lable_code = iris_data.iloc[:, 4].map(lable_mapping).tolist()
    # 数据集预处理，以花萼面积为横坐标，以花瓣面积做纵坐标
    hua_e = X[:, 0] * X[:, 1]
    hua_ban = X[:, 2] * X[:, 3]
    k = 3
    b = random_point(k)
    ceshi_hua_e = [hua_e[i] for i in range(len(hua_e)) if (i in b)]
    ceshi_hua_ban = [hua_ban[i] for i in range(len(hua_ban)) if (i in b)]
    ceshi_lable = [lable[i] for i in range(len(lable)) if (i in b)]
    x = hua_e
    y = hua_ban
    x0 = ceshi_hua_e
    y0 = ceshi_hua_ban
    n = 0
    ds = get_distance(x, y, x0, y0, k)
    temp = k_means(ds, x)
    temp1 = error_distance(x, y, x0, y0, k)
    n = n + 1
    center = cent(temp)
    x0 = center[0]
    y0 = center[1]
    ds = get_distance(x, y, x0, y0, k)
    temp = k_means(ds, x)
    temp2 = error_distance(x, y, x0, y0, k)
    n = n + 1
    # 判断迭代是否继续
    while np.abs(temp2 - temp1) != 0:
        temp1 = temp2
        center = cent(temp)
        x0 = center[0]
        y0 = center[1]
        ds = get_distance(x, y, x0, y0, k)
        temp = k_means(ds, x)
        temp2 = error_distance(x, y, x0, y0, k)
        n = n + 1
        print(n, temp2)
    print("迭代次数: ", n)
    k_view()
    
    注释
    由于之前在人工智能实验做过该k-means实验，因此直接使用了之前实验的鸢尾花数据集，得出的散点图。
    
 实验四
 ====
 代码：
sigmoid.py：  # sigmoid函数
# coding:utf-8
import matplotlib.pyplot as plt
import os
import numpy as np


def sigmoid():
    # 采样
    x = np.linspace(-10, 10, 500)
    y = 1.0 / (1 + np.exp(-x))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(title="sigmoid", xlabel="x", ylabel="y=f(x)")
    # 设置线宽和颜色
    ax.plot(x, y, linewidth=2.0, color="blue")
    # 保存图片到磁盘
    plt.savefig("./sigmoid.png", format="png")
    # 显示
    plt.show()

if __name__ == "__main__":
sigmoid()

Gradient Descent.py：  # 梯度下降
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
import numpy as np
import pandas as pd
# 加载数据集
X = []
Y = []
cancer = np.array(pd.read_csv('data.csv'))
for data in cancer:
  X.append(data[0:2])  # 样本
  Y.append(data[2])  # 类别
# 划分数据集
X_trainer = X
Y_trainer = Y
lr = linear_model.LogisticRegression()
lr.fit(X_trainer, Y_trainer)  # 训练
result = int(lr.predict([[2, 6]]))  # 测试,对(2,6)的结果进行预测
print('预测结果:', result)

注释
用Python完成，实验用到了sigmoid函数和逻辑回归算法。将实验三.2中的样例数据用聚类的结果打标签{0，1}，并用逻辑回归模型拟合。
画出了sigmoid函数
设计梯度下降算法，实现逻辑回归模型的学习过程。
根据给定数据（实验三.2），用梯度下降算法进行数据拟合，并用学习好的模型对(2,6)分类。
