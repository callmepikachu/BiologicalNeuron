# BiologicalNeuron - 生物神经元多模态情感分析项目

基于生物神经元机制的多模态情感分析模型，结合文本和图像特征进行情感分类。

## 📋 项目简介

本项目实现了一个受生物神经元启发的深度学习模型，用于多模态情感分析任务。模型模拟了生物神经元的顶树突和基底树突机制，包括：

- **顶树突机制**：处理全局特征，实现局部协作
- **基底树突机制**：基于Hebbian学习处理细节特征
- **NMDA样门控**：模拟神经元的门控机制
- **多神经元层**：由多个生物神经元组成的网络层

## 🚀 主要功能

### 1. 生物神经元模型
- 实现单个生物神经元（`BiologicalNeuron`）
- 支持多神经元层（`BiologicalNeuronLayer`）
- 模拟生物神经元的树突机制和门控功能

### 2. 多模态情感分析
- 支持MVSA-Single数据集
- 结合BERT文本特征和ResNet图像特征
- 三分类情感分析（负面、中性、正面）

### 3. 阅读理解任务
- 支持RACE数据集
- 多选项阅读理解任务
- 基于生物神经元的文本理解模型

### 4. 模型分析工具
- 神经元激活模式分析
- 特征重要性分析
- 神经元专业化分析
- 权重分布可视化

## 📁 项目结构

```
BiologicalNeuron/
├── src/                          # 核心源代码
│   ├── models/                   # 模型定义
│   │   ├── biological_neuron.py  # 生物神经元实现
│   │   └── biological_text_model.py # 文本模型
│   ├── data/                     # 数据处理
│   │   ├── mvsa_dataloader.py    # MVSA数据加载器
│   │   └── race.py              # RACE数据处理器
│   ├── config.py                # 配置文件
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   └── utils.py                 # 工具函数
├── task/                        # 任务脚本
│   ├── train_mvsa.py           # MVSA训练
│   ├── train_mvsa_extended.py  # 扩展训练
│   ├── analyze_neurons.py      # 神经元分析
│   ├── hyperparameter_tuning.py # 超参数调优
│   └── sota_research.py        # SOTA研究工具
├── data/                        # 数据集
│   ├── MVSA_Single/            # MVSA-Single数据集
│   └── RACE/                   # RACE数据集
├── results/                     # 实验结果
├── checkpoints/                 # 模型检查点
└── README.md                   # 项目说明
```

## 🛠️ 安装说明

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA/MPS支持（可选，用于GPU加速）

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd BiologicalNeuron
```

2. 安装依赖
```bash
pip install -r src/requirements.txt
```

3. 下载数据集
```bash
# 下载RACE数据集
bash src/download_race.sh

# 下载MVSA-Single数据集（需要手动下载）
# 将数据集解压到 data/MVSA_Single/ 目录
```

## 📖 使用方法

### 1. MVSA-Single多模态情感分析

#### 特征提取
```bash
cd task
python extract_mvsa_features.py
```

#### 训练模型
```bash
python train_mvsa.py
```

#### 扩展训练（更多epoch）
```bash
python train_mvsa_extended.py
```

#### 超参数调优
```bash
python hyperparameter_tuning.py
```

### 2. RACE阅读理解任务

#### 训练模型
```bash
cd src
python train.py
```

#### 评估模型
```bash
python evaluate.py
```

### 3. 神经元分析

#### 分析神经元激活模式
```bash
cd task
python analyze_neurons.py
```

#### 快速参数调优
```bash
python quick_tuning.py
```

## 📊 实验结果

### MVSA-Single多模态情感分析
- **最佳准确率**: 62.42%
- **模型配置**: 32个神经元，学习率0.001，批量大小32
- **训练时间**: 约50个epoch

### 神经元分析结果
- 发现神经元存在专业化和通用化分工
- 部分神经元对特定情感类别敏感
- 神经元激活模式具有可解释性

### 与SOTA对比
- 当前模型在准确率上接近SOTA水平
- 优势在于模型的可解释性和轻量级设计
- 适合用于研究生物启发的神经网络机制

## 🔬 技术特点

### 生物神经元机制
1. **顶树突处理**：处理全局特征，实现局部协作
2. **基底树突处理**：基于Hebbian学习处理细节特征
3. **NMDA门控**：模拟神经元的门控机制
4. **动作电位**：模拟神经元的输出机制

### 多模态融合
- 文本特征：使用BERT提取语义特征
- 图像特征：使用ResNet提取视觉特征
- 特征融合：通过生物神经元层进行融合

### 模型优势
- **可解释性**：神经元激活模式可分析
- **轻量级**：相比传统Transformer更轻量
- **生物启发**：基于真实神经元机制设计
- **多模态**：支持文本和图像联合分析

## 📈 性能优化

### 超参数调优
- 使用Optuna进行自动化超参数搜索
- 支持快速参数调优模式
- 实现了学习率调度和早停机制

### 训练优化
- 支持分块训练，处理大规模数据
- 实现了特征缓存机制
- 支持多GPU训练

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 开发环境设置
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。

## 🙏 致谢

- 感谢MVSA-Single和RACE数据集的提供者
- 感谢PyTorch和Transformers社区的支持
- 感谢所有为项目做出贡献的开发者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]

---

**注意**: 本项目主要用于研究目的，展示了生物启发的神经网络设计思路。在实际应用中，建议根据具体需求进行适当的调整和优化。
