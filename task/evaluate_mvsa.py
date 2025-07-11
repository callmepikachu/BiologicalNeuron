import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.biological_neuron import BiologicalNeuronLayer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_features(data_dir, split):
    """加载预提取的特征"""
    features = torch.load(os.path.join(data_dir, f'{split}_features.pt'))
    return features['x_top'], features['x_base'], features['labels']

def evaluate_mvsa_model(data_dir='data/MVSA_Single', model_path=None):
    """评估MVSA-Single多模态情感分析模型"""
    # 设备检测：优先MPS，然后CUDA，最后CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用设备: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用设备: CUDA")
    else:
        device = torch.device('cpu')
        print("使用设备: CPU")
    
    # 加载特征
    print('加载测试特征...')
    test_text, test_img, test_labels = load_features(data_dir, 'test')
    print(f'测试集: {test_text.shape}, {test_img.shape}, {test_labels.shape}')
    
    # 创建数据加载器
    test_dataset = TensorDataset(test_text, test_img, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 模型参数
    text_dim = test_text.shape[1]  # BERT特征维度
    img_dim = test_img.shape[1]    # ResNet特征维度
    num_classes = 3  # negative, neutral, positive
    
    # 创建模型
    model = BiologicalNeuronLayer(
        input_dim_top=text_dim,
        input_dim_base=img_dim,
        num_neurons=64  # 输出64维特征
    ).to(device)
    
    # 分类头
    classifier = nn.Sequential(
        nn.Linear(64, 128),  # BiologicalNeuronLayer输出是64维
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    ).to(device)
    
    # 加载模型权重
    if model_path is None:
        model_path = os.path.join(data_dir, 'best_mvsa_model.pth')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print(f'加载模型: {model_path}')
        if 'accuracy' in checkpoint:
            print(f'训练时最佳准确率: {checkpoint["accuracy"]:.4f}')
    else:
        print(f'模型文件不存在: {model_path}')
        return
    
    # 评估
    model.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_text, batch_img, batch_labels in test_loader:
            batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
            
            bio_output = model(batch_text, batch_img)
            outputs = classifier(bio_output)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f'\n=== MVSA-Single 多模态情感分析评估结果 ===')
    print(f'测试准确率: {accuracy:.4f}')
    
    # 分类报告
    print('\n分类报告:')
    print(classification_report(all_labels, all_preds, 
                               target_names=['negative', 'neutral', 'positive'],
                               digits=4))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title('MVSA-Single Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('results/mvsa_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 各类别准确率
    class_names = ['negative', 'neutral', 'positive']
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracies, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    plt.title('Class-wise Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # 在柱状图上显示数值
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.savefig('results/mvsa_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 预测概率分布
    all_probs = np.array(all_probs)
    plt.figure(figsize=(15, 5))
    
    for i, class_name in enumerate(class_names):
        plt.subplot(1, 3, i+1)
        plt.hist(all_probs[:, i], bins=20, alpha=0.7, color=['#ff6b6b', '#4ecdc4', '#45b7d1'][i])
        plt.title(f'{class_name} Prediction Probability Distribution')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Sample Count')
    
    plt.tight_layout()
    plt.savefig('results/mvsa_probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, all_preds, all_labels, all_probs

if __name__ == '__main__':
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 评估模型
    accuracy, preds, labels, probs = evaluate_mvsa_model(
        data_dir='data/MVSA_Single'
    )
    
    print(f'\n评估完成！最终测试准确率: {accuracy:.4f}') 