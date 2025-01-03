# -*- coding:utf-8 -*-
import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化参数
# 正如我们在构造nn.Linear时指定输入和输出尺寸一样， 现在我们能直接访问参数以设定它们的初始值。
# 我们通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数。
# 我们还可以使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

print(net[0].weight.data)
# tensor([[-0.0018, -0.0165]])

# 损失函数
# 计算均方误差使用的是MSELoss类，也称为平方L2范数。默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()

# 优化算法
# PyTorch在optim模块
# 当我们实例化一个SGD实例时，我们要指定优化的参数
# （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。
# 小批量随机梯度下降只需要设置lr值，这里设置为0.03。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
# epoch 1, loss 0.000180
# epoch 2, loss 0.000100
# epoch 3, loss 0.000101

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
# w的估计误差： tensor([-0.0002, -0.0010])
# b的估计误差： tensor([-2.8610e-06])