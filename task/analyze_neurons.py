import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.biological_neuron import BiologicalNeuronLayer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def load_features(data_dir, split):
    """加载预提取的特征"""
    features = torch.load(os.path.join(data_dir, f'{split}_features.pt'))
    return features['x_top'], features['x_base'], features['labels']

def load_best_model():
    """加载最佳模型"""
    # 设备检测
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 加载数据获取维度
    train_text, train_img, _ = load_features('data/MVSA_Single', 'train')
    text_dim = train_text.shape[1]
    img_dim = train_img.shape[1]
    
    # 创建模型
    model = BiologicalNeuronLayer(
        input_dim_top=text_dim,
        input_dim_base=img_dim,
        num_neurons=32
    ).to(device)
    
    # 分类头
    classifier = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 3)
    ).to(device)
    
    # 加载权重
    model_path = 'data/MVSA_Single/best_quick_tuned_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print(f"加载模型: {model_path}")
        print(f"模型准确率: {checkpoint['accuracy']:.4f}")
    else:
        print(f"模型文件不存在: {model_path}")
        return None, None, device
    
    return model, classifier, device

def analyze_neuron_weights(model):
    """分析神经元权重分布"""
    print("\n=== 神经元权重分析 ===")
    
    # 提取权重
    top_weights = []  # 文本权重
    base_weights = []  # 图像权重
    gating_weights = []  # 门控权重
    
    for neuron in model.neurons:
        top_weights.append(neuron.top_weights.detach().cpu().numpy())
        base_weights.append(neuron.base_weights.detach().cpu().numpy())
        gating_weights.append(neuron.gating_weights.detach().cpu().numpy())
    
    top_weights = np.array(top_weights)  # [32, 768]
    base_weights = np.array(base_weights)  # [32, 512]
    gating_weights = np.array(gating_weights)  # [32, 768]
    
    # 权重统计
    print(f"文本权重形状: {top_weights.shape}")
    print(f"图像权重形状: {base_weights.shape}")
    print(f"门控权重形状: {gating_weights.shape}")
    
    # 权重分布可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 文本权重分布
    axes[0, 0].hist(top_weights.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Text Weights Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # 图像权重分布
    axes[0, 1].hist(base_weights.flatten(), bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Image Weights Distribution')
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # 门控权重分布
    axes[0, 2].hist(gating_weights.flatten(), bins=50, alpha=0.7, color='red')
    axes[0, 2].set_title('Gating Weights Distribution')
    axes[0, 2].set_xlabel('Weight Value')
    axes[0, 2].set_ylabel('Frequency')
    
    # 神经元权重强度热图
    sns.heatmap(top_weights, ax=axes[1, 0], cmap='RdBu_r', center=0)
    axes[1, 0].set_title('Text Weights Heatmap')
    axes[1, 0].set_xlabel('Text Feature Dimension')
    axes[1, 0].set_ylabel('Neuron Index')
    
    sns.heatmap(base_weights, ax=axes[1, 1], cmap='RdBu_r', center=0)
    axes[1, 1].set_title('Image Weights Heatmap')
    axes[1, 1].set_xlabel('Image Feature Dimension')
    axes[1, 1].set_ylabel('Neuron Index')
    
    # 权重强度对比
    top_strength = np.linalg.norm(top_weights, axis=1)
    base_strength = np.linalg.norm(base_weights, axis=1)
    
    axes[1, 2].bar(range(32), top_strength, alpha=0.7, label='Text', color='blue')
    axes[1, 2].bar(range(32), base_strength, alpha=0.7, label='Image', color='green')
    axes[1, 2].set_title('Weight Strength by Neuron')
    axes[1, 2].set_xlabel('Neuron Index')
    axes[1, 2].set_ylabel('Weight Strength')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('results/neuron_weights_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_weights, base_weights, gating_weights

def analyze_neuron_activations(model, device):
    """分析神经元激活模式"""
    print("\n=== 神经元激活模式分析 ===")
    
    # 加载测试数据
    test_text, test_img, test_labels = load_features('data/MVSA_Single', 'test')
    test_dataset = TensorDataset(test_text, test_img, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 收集激活数据
    all_activations = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_text, batch_img, batch_labels in test_loader:
            batch_text, batch_img = batch_text.to(device), batch_img.to(device)
            
            # 获取每个神经元的激活
            activations = []
            for neuron in model.neurons:
                # 计算顶树突激活
                top_input = batch_text * neuron.top_weights.unsqueeze(0)
                local_activity = top_input.sum(dim=1) + neuron.top_bias
                
                # 计算门控信号
                gate_signal = (batch_text * neuron.gating_weights.unsqueeze(0)).sum(dim=1)
                gate_open = torch.sigmoid(gate_signal)
                top_contribution = local_activity * gate_open
                
                # 计算基底树突激活
                base_contribution = (batch_img * neuron.base_weights.unsqueeze(0)).sum(dim=1) + neuron.base_bias
                
                # 整合输出
                combined = torch.stack([top_contribution, base_contribution], dim=1)
                output = torch.sigmoid((combined @ neuron.output_weight) + neuron.output_bias)
                activations.append(output)
            
            # 堆叠所有神经元的激活 [batch_size, 32]
            batch_activations = torch.stack(activations, dim=1)
            all_activations.append(batch_activations.cpu().numpy())
            all_labels.append(batch_labels.numpy())
    
    all_activations = np.concatenate(all_activations, axis=0)  # [N, 32]
    all_labels = np.concatenate(all_labels, axis=0)  # [N]
    
    print(f"激活数据形状: {all_activations.shape}")
    print(f"标签数据形状: {all_labels.shape}")
    
    # 激活模式可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 神经元激活热图
    sns.heatmap(all_activations.T, ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title('Neuron Activation Heatmap')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Neuron Index')
    
    # 2. 各类别的平均激活
    label_names = ['negative', 'neutral', 'positive']
    avg_activations = []
    for label in range(3):
        mask = all_labels == label
        avg_act = all_activations[mask].mean(axis=0)
        avg_activations.append(avg_act)
    
    avg_activations = np.array(avg_activations)  # [3, 32]
    
    sns.heatmap(avg_activations, ax=axes[0, 1], cmap='viridis', 
                xticklabels=range(32), yticklabels=label_names)
    axes[0, 1].set_title('Average Activation by Class')
    axes[0, 1].set_xlabel('Neuron Index')
    axes[0, 1].set_ylabel('Class')
    
    # 3. 神经元激活分布
    axes[1, 0].hist(all_activations.flatten(), bins=50, alpha=0.7, color='purple')
    axes[1, 0].set_title('Neuron Activation Distribution')
    axes[1, 0].set_xlabel('Activation Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. 各类别激活强度对比
    for i, label_name in enumerate(label_names):
        axes[1, 1].plot(avg_activations[i], label=label_name, marker='o')
    axes[1, 1].set_title('Average Activation by Neuron and Class')
    axes[1, 1].set_xlabel('Neuron Index')
    axes[1, 1].set_ylabel('Average Activation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/neuron_activation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_activations, all_labels

def analyze_neuron_specialization(all_activations, all_labels):
    """分析神经元专业化程度"""
    print("\n=== 神经元专业化分析 ===")
    
    # 计算每个神经元对各类别的响应差异
    label_names = ['negative', 'neutral', 'positive']
    specialization_scores = []
    
    for neuron_idx in range(32):
        neuron_activations = all_activations[:, neuron_idx]
        
        # 计算各类别的平均激活
        class_means = []
        for label in range(3):
            mask = all_labels == label
            class_mean = neuron_activations[mask].mean()
            class_means.append(class_mean)
        
        # 计算专业化分数（类别间方差）
        specialization_score = np.var(class_means)
        specialization_scores.append(specialization_score)
    
    # 找出最专业化的神经元
    top_specialized = np.argsort(specialization_scores)[-10:]  # 前10个
    least_specialized = np.argsort(specialization_scores)[:10]  # 后10个
    
    print("最专业化的神经元 (对特定类别响应强烈):")
    for i, neuron_idx in enumerate(top_specialized):
        print(f"  神经元 {neuron_idx}: 专业化分数 {specialization_scores[neuron_idx]:.4f}")
    
    print("\n最通用的神经元 (对所有类别响应相似):")
    for i, neuron_idx in enumerate(least_specialized):
        print(f"  神经元 {neuron_idx}: 专业化分数 {specialization_scores[neuron_idx]:.4f}")
    
    # 可视化专业化分数
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(32), specialization_scores, color='orange', alpha=0.7)
    plt.title('Neuron Specialization Scores')
    plt.xlabel('Neuron Index')
    plt.ylabel('Specialization Score')
    plt.axhline(y=np.mean(specialization_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(specialization_scores):.4f}')
    plt.legend()
    
    # 专业化分数分布
    plt.subplot(1, 2, 2)
    plt.hist(specialization_scores, bins=15, alpha=0.7, color='orange')
    plt.title('Specialization Score Distribution')
    plt.xlabel('Specialization Score')
    plt.ylabel('Number of Neurons')
    
    plt.tight_layout()
    plt.savefig('results/neuron_specialization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return specialization_scores

def analyze_feature_importance(model, all_activations, all_labels):
    """分析特征重要性"""
    print("\n=== 特征重要性分析 ===")
    
    # 计算每个神经元对最终分类的贡献
    from sklearn.linear_model import LogisticRegression
    
    # 使用神经元激活预测标签
    clf = LogisticRegression(random_state=42)
    clf.fit(all_activations, all_labels)
    
    # 特征重要性（系数绝对值）
    feature_importance = np.abs(clf.coef_)  # [3, 32]
    
    # 可视化特征重要性
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 各类别的特征重要性
    label_names = ['negative', 'neutral', 'positive']
    for i, label_name in enumerate(label_names):
        axes[0, 0].bar(range(32), feature_importance[i], alpha=0.7, label=label_name)
    axes[0, 0].set_title('Feature Importance by Class')
    axes[0, 0].set_xlabel('Neuron Index')
    axes[0, 0].set_ylabel('Feature Importance')
    axes[0, 0].legend()
    
    # 2. 总体特征重要性
    overall_importance = feature_importance.mean(axis=0)
    axes[0, 1].bar(range(32), overall_importance, color='purple', alpha=0.7)
    axes[0, 1].set_title('Overall Feature Importance')
    axes[0, 1].set_xlabel('Neuron Index')
    axes[0, 1].set_ylabel('Feature Importance')
    
    # 3. 特征重要性热图
    sns.heatmap(feature_importance, ax=axes[1, 0], cmap='viridis',
                xticklabels=range(32), yticklabels=label_names)
    axes[1, 0].set_title('Feature Importance Heatmap')
    axes[1, 0].set_xlabel('Neuron Index')
    axes[1, 0].set_ylabel('Class')
    
    # 4. 最重要的神经元
    top_neurons = np.argsort(overall_importance)[-10:]
    axes[1, 1].bar(range(10), overall_importance[top_neurons], color='red', alpha=0.7)
    axes[1, 1].set_title('Top 10 Most Important Neurons')
    axes[1, 1].set_xlabel('Rank')
    axes[1, 1].set_ylabel('Feature Importance')
    axes[1, 1].set_xticks(range(10))
    axes[1, 1].set_xticklabels([f'N{idx}' for idx in top_neurons])
    
    plt.tight_layout()
    plt.savefig('results/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出最重要的神经元
    print("最重要的神经元 (对分类贡献最大):")
    for i, neuron_idx in enumerate(top_neurons[::-1]):  # 从最重要到最不重要
        print(f"  {i+1}. 神经元 {neuron_idx}: 重要性 {overall_importance[neuron_idx]:.4f}")
    
    return feature_importance, overall_importance

def main():
    """主函数"""
    print("开始分析32个神经元的激活模式...")
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 加载模型
    model, classifier, device = load_best_model()
    if model is None:
        return
    
    # 1. 分析神经元权重
    top_weights, base_weights, gating_weights = analyze_neuron_weights(model)
    
    # 2. 分析神经元激活模式
    all_activations, all_labels = analyze_neuron_activations(model, device)
    
    # 3. 分析神经元专业化
    specialization_scores = analyze_neuron_specialization(all_activations, all_labels)
    
    # 4. 分析特征重要性
    feature_importance, overall_importance = analyze_feature_importance(model, all_activations, all_labels)
    
    print("\n=== 分析完成 ===")
    print("所有分析结果已保存到 results/ 目录")
    print("生成的文件:")
    print("- neuron_weights_analysis.png")
    print("- neuron_activation_analysis.png") 
    print("- neuron_specialization_analysis.png")
    print("- feature_importance_analysis.png")

if __name__ == '__main__':
    main() 