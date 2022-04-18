#开发时间：2022/4/18 15:12
import torch
from torch import nn
from torch.nn import functional as F

#nn.Sequential 与class(nn.Module)的混合使用

# net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

x=torch.rand(2,20)
# print(net(x))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden=nn.Linear(20,256)
        self.out=nn.Linear(256,10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))


# net=MLP()
# print(net(x))

# 自定义Sequential
class MySequential(nn.Module):
    # *args:list of arguments
    def __init__(self,*args):
        super(MySequential, self).__init__()
        for blocks in args:
            # _modules[]: dict of order
            self._modules[blocks]=blocks


    def forward(self,X):
        for block in self._modules.values():
            X=block(X)
        return X

# net=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
# print(net(x))

#继承nn.Module更灵活的定义参数、做前向计算
class FixedHiddenModule(nn.Module):
    def __init__(self):
        super(FixedHiddenModule, self).__init__()
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        self.linear=nn.Linear(20,20)

    def forward(self,X):
        X=self.linear(X)
        X=F.relu(torch.mm(X,self.rand_weight)+1)
        X=self.linear(X)
        while X.abs().sum()>1:
            X/=2
        return X.sum()
# net = FixedHiddenModule()
# print(net(x))


#nn.Sequential 与class(nn.Module)的混合使用
class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear=nn.Linear(32,16)

    def forward(self,X):
        return self.linear(self.net(X))

chimera=nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenModule())
print(chimera(x))



