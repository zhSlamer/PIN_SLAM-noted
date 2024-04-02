#!/usr/bin/env python3
# @file      decoder.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad

from utils.config import Config

# MLP 多层感知器（Multilayer Perceptron） 它由一个输入层、一个或多个隐藏层以及一个输出层组成。
# hidden_dim：隐藏层维数， hidden_level：层数， out_dim：输出维数
class Decoder(nn.Module):
    def __init__(self, config: Config, hidden_dim, hidden_level, out_dim, is_time_conditioned = False): 
        
        super().__init__()
    
        self.out_dim = out_dim
        # 加入偏置项
        bias_on = config.mlp_bias_on

        self.use_leaky_relu = False
        # pos_encoding_band: int = 0 # if 0, without encoding
        # pos_input_dim: int = 3
        # pos_encoding_base: int = 2
        self.num_bands = config.pos_encoding_band
        self.dimensionality = config.pos_input_dim
        # 对输入的位置维度编码
        if config.use_gaussian_pe:
            position_dim = config.pos_input_dim + 2 * config.pos_encoding_band
        else:
            position_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1)
        
        # 潜在特征向量
        feature_dim = config.feature_dim
        input_layer_count = feature_dim + position_dim  # 8 + 3
        # 时间输入维度
        if is_time_conditioned:
            input_layer_count += 1

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        # 创建一个空列表，用于储存MLP的各个层
        layers = []
        # 遍历隐藏层数量 hidden_level， 为每一层创建一个线性层， hidden_level为1， 故0，1共两个隐藏层
        for i in range(hidden_level):
            # 对于第一层，创建一个线性层，其输入大小为 input_layer_count，输出大小为 hidden_dim。
            # 这里的 nn.Linear 表示一个线性层，第一个参数是输入大小，第二个参数是输出大小，第三个参数是是否使用偏置（根据 bias_on 决定）。
            if i == 0:
                layers.append(nn.Linear(input_layer_count, hidden_dim, bias_on))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias_on))
        # 将创建的所有线性层存储在 nn.ModuleList 中。
        # nn.ModuleList 用于将多个模块存储为列表，并将它们注册为当前模块的子模块，以便在模型参数优化时能够自动进行管理。
        self.layers = nn.ModuleList(layers)
        # 创建最后一层的线性层，其输入大小为 hidden_dim，输出大小为 out_dim。这一层用于将隐藏层的输出映射到最终的 SDF 预测结果。
        self.lout = nn.Linear(hidden_dim, out_dim, bias_on)

        # 根据配置文件中指定的主要损失类型，设置 SDF 的缩放系数 self.sdf_scale。如果主要损失类型是二分类交叉熵损失（'bce'），则使用指定的缩放系数，否则设为 1。
        if config.main_loss_type == 'bce':
            # logistic_gaussian_ratio: float = 0.55 # the factor ratio for approximize a Gaussian distribution using the derivative of logistic function
            # sigma_sigmoid_m: float = 0.1 # better to be set according to the noise level
            self.sdf_scale = config.logistic_gaussian_ratio*config.sigma_sigmoid_m
        else: # l1, l2 or zhong loss
            self.sdf_scale = 1.
        # 将模型移动到指定的设备上，根据配置文件中的 config.device。这是为了在 GPU 上进行计算加速，如果没有指定设备，则使用默认的设备。
        self.to(config.device)
        # torch.cuda.empty_cache()

    def forward(self, feature):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sdf(feature)
        return output

    # 这段代码实现了一个前向传播函数 sdf，用于对输入特征进行处理，并最终预测 Signed Distance Field（SDF）的值。
    # 通过 MLP 的多层处理，模型能够学习并预测复杂几何形状的 SDF，其中使用了 ReLU 或 Leaky ReLU 激活函数，并对最终输出进行了缩放操作。
    # predict the sdf (opposite sign to the actual sdf)
    # unit is already m
    # 输入特征 11 维向量
    def sdf(self, features):
        # 遍历MLP所有层 k 为层的索引，l表示当前层对象
        for k, l in enumerate(self.layers):
            if k == 0:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(features))
                else:
                    h = F.relu(l(features))
            else:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(h))
                else:
                    h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        out *= self.sdf_scale
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out
    
    
    def time_conditionded_sdf(self, features, ts):

        # print(ts.shape)
        nn_k = features.shape[1]
        ts_nn_k = ts.repeat(nn_k).view(-1, nn_k, 1)
        time_conditioned_feature = torch.cat((features, ts_nn_k), dim=-1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(time_conditioned_feature))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        out *= self.sdf_scale
        # linear (feature_dim + 1 -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

    # predict the occupancy probability
    def occupancy(self, features):
        out = torch.sigmoid(self.sdf(features)/-self.sdf_scale)  # to [0, 1]
        return out

    # predict the probabilty of each semantic label
    def sem_label_prob(self, features):
        for k, l in enumerate(self.layers):
            if k == 0:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(features))
                else:
                    h = F.relu(l(features))
            else:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(h))
                else:
                    h = F.relu(l(h))

        out = F.log_softmax(self.lout(h), dim=-1)
        return out

    def sem_label(self, features):
        out = torch.argmax(self.sem_label_prob(features), dim=1)
        return out
    
    def regress_color(self, features):
        for k, l in enumerate(self.layers):
            if k == 0:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(features))
                else:
                    h = F.relu(l(features))
            else:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(h))
                else:
                    h = F.relu(l(h))

        out = torch.clamp(self.lout(h), 0., 1.)
        # out = torch.sigmoid(self.lout(h))
        # print(out)
        return out
