# -*- coding:utf-8 -*-
import torch

x = torch.arange(4.0)
print(x)
# tensor([0., 1., 2., 3.])

# 一个标量函数关于向量x的梯度是向量，并且与x具有相同的形状。
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)  # 默认值是None
# None

y = 2 * torch.dot(x, x)
print(y)
# tensor(28., grad_fn=<MulBackward0>)

# x是一个长度为4的向量，计算x和x的点积，得到了我们赋值给y的标量输出。 接下来，通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
y.backward()
print(x.grad)
# tensor([ 0.,  4.,  8., 12.])

# 函数y = 2x^Tx关于x的梯度应为4x。 让我们快速验证这个梯度是否计算正确。
print(x.grad == 4 * x)
# tensor([True, True, True, True])

# 现在让我们计算x的另一个函数。
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)
# tensor([1., 1., 1., 1.])

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
y.sum().backward()  # 等价于y.backward(torch.ones(len(x)))
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad)
print(x.grad == u)


x.grad.zero_()
y.sum().backward()
print(x.grad)
print(x.grad == 2 * x)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)
print(a.grad == d / a)