import pandas as pd



srdata = pd.read_csv('txt表1.csv', encoding='gbk')
srdata['ID']=srdata['ID'].apply(lambda x:'20200'+str(x) if x<=9 else "202"+str(x))


with open('txt表1.txt', 'w', encoding='UTF-8') as outfile:

    srdata.to_string(outfile,index=None)