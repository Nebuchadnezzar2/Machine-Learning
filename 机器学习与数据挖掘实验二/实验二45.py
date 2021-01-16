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


