import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def loss_rate(x):  # x is DataFrmae type
    table = ((x.isnull().sum())/x.shape[0]).sort_values(ascending=False).map(lambda i:"{:.5%}".format(i))
    return table

def vector_distances(v, x):
    """
    计算向量v与矩阵X中每个样本向量之间的距离
    
    参数:
    v (array-like): 输入向量，形状为(n,)
    x (array-like): 输入矩阵，形状为(m, n)，每行表示一个样本向量
    
    返回:
    distances (array-like): 距离数组，形状为(m,)，表示向量v与矩阵X中每个样本向量之间的距离
    """
    
    diff = x - v  # 计算差值
    squared_diff = diff ** 2  # 计算平方差值
    sum_squared_diff = np.sum(squared_diff, axis=1)  # 沿着列方向求和
    distances = np.sqrt(sum_squared_diff)  # 开方
    return distances