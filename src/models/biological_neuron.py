import torch
import torch.nn as nn

class BiologicalNeuron(nn.Module):
    def __init__(self, input_dim_top, input_dim_base):
        super(BiologicalNeuron, self).__init__()
        # 顶树突权重（局部协作）
        self.top_weights = nn.Parameter(torch.randn(input_dim_top))
        self.top_bias = nn.Parameter(torch.zeros(1))
        # 基底树突权重（Hebbian机制）
        self.base_weights = nn.Parameter(torch.randn(input_dim_base))
        self.base_bias = nn.Parameter(torch.zeros(1))
        # NMDA样门控参数
        self.gating_weights = nn.Parameter(torch.randn(input_dim_top))
        self.gate_activation = torch.sigmoid
        # 输出整合参数
        self.output_weight = nn.Parameter(torch.randn(2))
        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_top, x_base, output_required=False):
        """
        x_top: [batch_size, input_dim_top]，顶树突输入（如全局特征）
        x_base: [batch_size, input_dim_base]，基底树突输入（如细节特征）
        output_required: 是否产生动作电位（模拟反向传播/强化）
        """
        # 顶树突机制
        top_input = x_top * self.top_weights.unsqueeze(0)
        local_activity = top_input.sum(dim=1) + self.top_bias
        gate_signal = (x_top * self.gating_weights.unsqueeze(0)).sum(dim=1)
        gate_open = self.gate_activation(gate_signal)
        top_contribution = local_activity * gate_open
        # 基底树突机制（Hebbian）
        base_contribution = (x_base * self.base_weights.unsqueeze(0)).sum(dim=1) + self.base_bias
        if output_required:
            base_contribution += 0.1 * x_base.mean(dim=1)  # 模拟强化连接
        # 整合输出
        combined = torch.stack([top_contribution, base_contribution], dim=1)
        output = torch.sigmoid((combined @ self.output_weight) + self.output_bias)
        return output

class BiologicalNeuronLayer(nn.Module):
    def __init__(self, input_dim_top, input_dim_base, num_neurons):
        super(BiologicalNeuronLayer, self).__init__()
        # 多个神经元组成一层
        self.neurons = nn.ModuleList([
            BiologicalNeuron(input_dim_top, input_dim_base)
            for _ in range(num_neurons)
        ])

    def forward(self, x_top, x_base, output_required=False):
        # 并行计算每个神经元的输出
        outputs = [neuron(x_top, x_base, output_required) for neuron in self.neurons]
        return torch.stack(outputs, dim=1)  # [batch_size, num_neurons] 