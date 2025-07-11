import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.biological_neuron import BiologicalNeuronLayer
from sklearn.model_selection import train_test_split
import json
import time

def load_features(data_dir, split):
    """加载预提取的特征"""
    features = torch.load(os.path.join(data_dir, f'{split}_features.pt'))
    return features['x_top'], features['x_base'], features['labels']

def train_and_evaluate(params, device=None):
    """训练并评估模型"""
    # 设备检测：优先MPS，然后CUDA，最后CPU
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    # 加载数据
    train_text, train_img, train_labels = load_features('data/MVSA_Single', 'train')
    
    # 划分训练集和验证集
    train_text, val_text, train_img, val_img, train_labels, val_labels = train_test_split(
        train_text, train_img, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # 模型参数
    text_dim = train_text.shape[1]
    img_dim = train_img.shape[1]
    num_classes = 3
    
    # 创建模型
    model = BiologicalNeuronLayer(
        input_dim_top=text_dim,
        input_dim_base=img_dim,
        num_neurons=params['num_neurons']
    ).to(device)
    
    # 分类头
    classifier = nn.Sequential(
        nn.Linear(params['num_neurons'], params['hidden_dim']),
        nn.ReLU(),
        nn.Dropout(params['dropout_rate']),
        nn.Linear(params['hidden_dim'], num_classes)
    ).to(device)
    
    # 数据加载器
    train_dataset = TensorDataset(train_text, train_img, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    val_dataset = TensorDataset(val_text, val_img, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=params['learning_rate'])
    
    # 训练循环
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(params['epochs']):
        # 训练
        model.train()
        classifier.train()
        for batch_text, batch_img, batch_labels in train_loader:
            batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            bio_output = model(batch_text, batch_img)
            outputs = classifier(bio_output)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_text, batch_img, batch_labels in val_loader:
                batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
                bio_output = model(batch_text, batch_img)
                outputs = classifier(bio_output)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_acc = correct / total
        
        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    return best_val_acc

def quick_parameter_search():
    """快速参数搜索"""
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
    
    # 参数组合
    param_combinations = [
        # 基础配置
        {
            'name': '基础配置',
            'num_neurons': 64,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 30
        },
        # 更多神经元
        {
            'name': '更多神经元',
            'num_neurons': 128,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 30
        },
        # 更深的网络
        {
            'name': '更深的网络',
            'num_neurons': 64,
            'hidden_dim': 256,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 30
        },
        # 更高dropout
        {
            'name': '更高dropout',
            'num_neurons': 64,
            'hidden_dim': 128,
            'dropout_rate': 0.5,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 30
        },
        # 更小学习率
        {
            'name': '更小学习率',
            'num_neurons': 64,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'epochs': 30
        },
        # 更大batch size
        {
            'name': '更大batch size',
            'num_neurons': 64,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 30
        },
        # 组合优化
        {
            'name': '组合优化',
            'num_neurons': 96,
            'hidden_dim': 192,
            'dropout_rate': 0.4,
            'learning_rate': 0.0005,
            'batch_size': 48,
            'epochs': 40
        },
        # 轻量级配置
        {
            'name': '轻量级配置',
            'num_neurons': 32,
            'hidden_dim': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.002,
            'batch_size': 16,
            'epochs': 25
        }
    ]
    
    results = []
    
    print("开始快速参数搜索...")
    print(f"将测试 {len(param_combinations)} 种配置")
    
    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] 测试配置: {params['name']}")
        print(f"参数: {params}")
        
        start_time = time.time()
        val_acc = train_and_evaluate(params, device)
        end_time = time.time()
        
        result = {
            'name': params['name'],
            'params': params,
            'validation_accuracy': val_acc,
            'training_time': end_time - start_time
        }
        results.append(result)
        
        print(f"验证准确率: {val_acc:.4f}")
        print(f"训练时间: {end_time - start_time:.2f}秒")
    
    # 排序结果
    results.sort(key=lambda x: x['validation_accuracy'], reverse=True)
    
    # 输出最佳结果
    print("\n" + "="*50)
    print("参数搜索结果 (按准确率排序):")
    print("="*50)
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['name']}: {result['validation_accuracy']:.4f}")
        print(f"   参数: {result['params']}")
        print(f"   训练时间: {result['training_time']:.2f}秒")
        print()
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    with open('results/quick_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 返回最佳参数
    best_result = results[0]
    print(f"最佳配置: {best_result['name']}")
    print(f"最佳验证准确率: {best_result['validation_accuracy']:.4f}")
    
    return best_result['params']

def train_final_model(best_params):
    """使用最佳参数训练最终模型"""
    print("\n使用最佳参数训练最终模型...")
    
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
    
    # 加载数据
    train_text, train_img, train_labels = load_features('data/MVSA_Single', 'train')
    test_text, test_img, test_labels = load_features('data/MVSA_Single', 'test')
    
    # 模型参数
    text_dim = train_text.shape[1]
    img_dim = train_img.shape[1]
    num_classes = 3
    
    # 创建模型
    model = BiologicalNeuronLayer(
        input_dim_top=text_dim,
        input_dim_base=img_dim,
        num_neurons=best_params['num_neurons']
    ).to(device)
    
    # 分类头
    classifier = nn.Sequential(
        nn.Linear(best_params['num_neurons'], best_params['hidden_dim']),
        nn.ReLU(),
        nn.Dropout(best_params['dropout_rate']),
        nn.Linear(best_params['hidden_dim'], num_classes)
    ).to(device)
    
    # 数据加载器
    train_dataset = TensorDataset(train_text, train_img, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    test_dataset = TensorDataset(test_text, test_img, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=best_params['learning_rate'])
    
    # 训练
    best_test_acc = 0.0
    train_losses = []
    test_accuracies = []
    
    for epoch in range(best_params['epochs']):
        # 训练
        model.train()
        classifier.train()
        total_loss = 0
        for batch_text, batch_img, batch_labels in train_loader:
            batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            bio_output = model(batch_text, batch_img)
            outputs = classifier(bio_output)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 测试
        model.eval()
        classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_text, batch_img, batch_labels in test_loader:
                batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
                bio_output = model(batch_text, batch_img)
                outputs = classifier(bio_output)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        test_acc = correct / total
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{best_params["epochs"]}: Loss={avg_loss:.4f}, Test Acc={test_acc:.4f}')
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'accuracy': test_acc,
                'best_params': best_params
            }, 'data/MVSA_Single/best_quick_tuned_model.pth')
    
    print(f"\n最终测试准确率: {best_test_acc:.4f}")
    return best_test_acc

if __name__ == '__main__':
    # 运行快速参数搜索
    best_params = quick_parameter_search()
    
    # 使用最佳参数训练最终模型
    final_acc = train_final_model(best_params)
    
    print(f"\n快速参数调优完成！")
    print(f"最终测试准确率: {final_acc:.4f}") 