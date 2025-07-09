import torch
import torch.nn as nn
from .biological_neuron import BiologicalNeuronLayer

class BiologicalTextModel(nn.Module):
    def __init__(self, input_dim_top, input_dim_base, hidden_size, output_size):
        super(BiologicalTextModel, self).__init__()
        # 第一层：输入为顶树突和基底树突特征
        self.layer1 = BiologicalNeuronLayer(input_dim_top, input_dim_base, hidden_size)
        # 第二层：输入为上一层输出
        self.layer2 = BiologicalNeuronLayer(hidden_size, hidden_size, output_size)
        # 分类器
        self.classifier = nn.Linear(output_size, 1)

    def forward(self, x_top, x_base):
        # 第一层神经元
        x = self.layer1(x_top, x_base, output_required=True)
        x = torch.relu(x)
        # 第二层神经元
        x = self.layer2(x, x, output_required=True)
        logits = self.classifier(x)
        return torch.sigmoid(logits) 