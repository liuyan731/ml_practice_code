# -*- coding:utf-8 -*-
"""
读取数据集
"""
# 创建一个csv数据集
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 使用pandas读取数据集
import pandas as pd
data = pd.read_csv(data_file)
print(data)
#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000

"""
处理缺失值
"""
# 为了处理缺失的数据，典型的方法包括插值法和删除法，
# 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。

# 通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。
# 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
# inputs = inputs.fillna(inputs.mean()) # 报错了
print(inputs)
#    NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN

# 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=float)
print(inputs)
#    NumRooms  Alley_Pave  Alley_nan
# 0       3.0         1.0        0.0
# 1       2.0         0.0        1.0
# 2       4.0         0.0        1.0
# 3       3.0         0.0        1.0

"""
转换为张量格式
"""
# 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。
import torch
X = torch.tensor(inputs.values)
y = torch.tensor(outputs.values)
print(X, y)
# tensor([[3., 1., 0.],
#         [2., 0., 1.],
#         [4., 0., 1.],
#         [3., 0., 1.]], dtype=torch.float64)
# tensor([127500, 106000, 178100, 140000])

"""
删除缺失值最多的列
"""
# 知道每列的nan数
nan_num = data.isna().sum(axis=0)
# 找到nan_num(series)中最大数的索引
nan_max_id = nan_num.idxmax()
# 删除nan最大的列
data = data.drop(columns=nan_max_id)
print(data)
#    NumRooms   Price
# 0       NaN  127500
# 1       2.0  106000
# 2       4.0  178100
# 3       NaN  140000
