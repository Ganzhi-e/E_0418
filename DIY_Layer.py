#开发时间：2022/4/18 16:39

import torch
import torch.nn.functional as F
from torch import nn

# 自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def forward(self,X):
        return X-X.mean()

layer=CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y=net(torch.rand((4,8)))
print(Y.mean())

# 自定义带参数的层
class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super(MyLinear, self).__init__()
        # nn.Parameter()用于赋予梯度属性 名字 !!!!!!!!!!
        self.weight=nn.Parameter(torch.randn((in_units,units)))
        self.bias=nn.Parameter(torch.zeros((units,)))

    def forward(self,X):
        # .data !!!!!!!!!!!!!!!!1111
        linear=torch.matmul(X,self.weight.data)+self.bias.data
        return F.relu(linear)

dense=MyLinear(5,3)
print(dense.weight)

print(dense(torch.rand((2,5))))

net=nn.Sequential(MyLinear(64,8),MyLinear(8,1))
print(net(torch.rand(2,64)))





