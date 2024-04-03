"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


# 这个函数是一个辅助函数，用于获取每一层的维度和激活函数。
def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """

    #如果args.act没有指定激活函数，那么激活函数就是恒等函数（即，不改变输入）；否则，从torch.nn.functional中获取激活函数。
    if not args.act:
        act = lambda x: x
    else:
        act_name = args.act
        if act_name.endswith('_'):  # Ensure non-inplace version is used
            act_name = act_name[:-1]
        act = getattr(F, act_name)
    #根据args.num_layers指定的层数，为每层设置激活函数和维度。
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class GraphConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features
    #前向传播方法接受一个输入元组，包含特征矩阵x和邻接矩阵adj
    def forward(self, input):
        x, adj = input
        #这些行应用线性变换和dropout。
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        # 根据邻接矩阵的稀疏性，选择稀疏或密集矩阵乘法。
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        #应用激活函数并返回输出及其邻接矩阵。


        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )

#这个类定义了一个带有dropout的简单线性层。
class Linear(Module):
    """
    Simple Linear layer with dropout.
    """
    #类似于GraphConvolution，这个初始化方法也设置了输入和输出特征、dropout率、激活函数和bias。
    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
    #前向传播方法应用线性变换、dropout和激活函数，然后返回输出。
    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out

#这个类定义了一个Fermi-Dirac解码器，用于计算基于距离的边概率。
class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""
    #初始化方法设置了Fermi-Dirac解码器的两个参数r和t。
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t
    #前向传播方法根据输入的距离计算边概率。
    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs






