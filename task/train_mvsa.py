import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
import torch.nn as nn
import torch.optim as optim
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

def train_mvsa_model(data_dir='data/MVSA_Single', epochs=50, lr=0.001, batch_size=32):
    """训练MVSA-Single多模态情感分析模型"""
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
    print('加载训练特征...')
    train_text, train_img, train_labels = load_features(data_dir, 'train')
    print('加载测试特征...')
    test_text, test_img, test_labels = load_features(data_dir, 'test')
    
    print(f'训练集: {train_text.shape}, {train_img.shape}, {train_labels.shape}')
    print(f'测试集: {test_text.shape}, {test_img.shape}, {test_labels.shape}')
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_text, train_img, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(test_text, test_img, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型参数
    text_dim = train_text.shape[1]  # BERT特征维度
    img_dim = train_img.shape[1]    # ResNet特征维度
    num_classes = 3  # negative, neutral, positive
    
    # 创建生物神经元模型
    model = BiologicalNeuronLayer(
        input_dim_top=text_dim,
        input_dim_base=img_dim,
        num_neurons=64  # 输出64维特征
    ).to(device)
    
    # 添加分类头
    classifier = nn.Sequential(
        nn.Linear(64, 128),  # BiologicalNeuronLayer输出是64维
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)
    
    # 训练记录
    train_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    
    print(f'开始训练，共{epochs}个epoch...')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch_text, batch_img, batch_labels in train_loader:
            batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            # 使用生物神经元模型
            bio_output = model(batch_text, batch_img)
            outputs = classifier(bio_output)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 测试阶段
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_text, batch_img, batch_labels in test_loader:
                batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
                bio_output = model(batch_text, batch_img)
                outputs = classifier(bio_output)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Test Acc={accuracy:.4f}')
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'accuracy': accuracy
            }, os.path.join(data_dir, 'best_mvsa_model.pth'))
            print(f'保存最佳模型，准确率: {accuracy:.4f}')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('results/mvsa_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 最终评估
    print(f'\n最终测试准确率: {best_accuracy:.4f}')
    print('\n分类报告:')
    print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('results/mvsa_confusion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, best_accuracy

if __name__ == '__main__':
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 开始训练
    model, best_acc = train_mvsa_model(
        data_dir='data/MVSA_Single',
        epochs=50,
        lr=0.001,
        batch_size=32
    )
    
    print(f'\n训练完成！最佳测试准确率: {best_acc:.4f}') 