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
