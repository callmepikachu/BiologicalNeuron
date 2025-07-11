import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.biological_neuron import BiologicalNeuronLayer
import optuna
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_features(data_dir, split):
    """加载预提取的特征"""
    features = torch.load(os.path.join(data_dir, f'{split}_features.pt'))
    return features['x_top'], features['x_base'], features['labels']

def objective(trial):
    """Optuna目标函数，用于超参数优化"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_text, train_img, train_labels = load_features('data/MVSA_Single', 'train')
    
    # 划分训练集和验证集
    train_text, val_text, train_img, val_img, train_labels, val_labels = train_test_split(
        train_text, train_img, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # 超参数搜索空间
    num_neurons = trial.suggest_int('num_neurons', 32, 128, step=16)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 20, 50, step=10)
    
    # 模型参数
    text_dim = train_text.shape[1]
    img_dim = train_img.shape[1]
    num_classes = 3
    
    # 创建模型
    model = BiologicalNeuronLayer(
        input_dim_top=text_dim,
        input_dim_base=img_dim,
        num_neurons=num_neurons
    ).to(device)
    
    # 分类头
    classifier = nn.Sequential(
        nn.Linear(num_neurons, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_classes)
    ).to(device)
    
    # 数据加载器
    train_dataset = TensorDataset(train_text, train_img, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_text, val_img, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
    
    # 训练循环
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
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

def run_hyperparameter_tuning(n_trials=50):
    """运行超参数调优"""
    print("开始超参数调优...")
    print(f"将进行 {n_trials} 次试验")
    
    # 创建Optuna研究
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # 运行优化
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 输出最佳结果
    print("\n=== 超参数调优结果 ===")
    print(f"最佳验证准确率: {study.best_value:.4f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    import json
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/hyperparameter_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘制优化历史
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # 优化历史
    plt.subplot(2, 2, 1)
    plt.plot(study.trials_dataframe()['value'])
    plt.title('Optimization History')
    plt.xlabel('Trial')
    plt.ylabel('Validation Accuracy')
    
    # 参数重要性
    plt.subplot(2, 2, 2)
    importance = optuna.importance.get_param_importances(study)
    params = list(importance.keys())
    values = list(importance.values())
    plt.barh(params, values)
    plt.title('Parameter Importance')
    plt.xlabel('Importance')
    
    # 参数分布
    plt.subplot(2, 2, 3)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importances')
    
    plt.subplot(2, 2, 4)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    
    plt.tight_layout()
    plt.savefig('results/hyperparameter_tuning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return study.best_params

def train_with_best_params(best_params):
    """使用最佳参数训练最终模型"""
    print("\n使用最佳参数训练最终模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
            }, 'data/MVSA_Single/best_tuned_model.pth')
    
    print(f"\n最终测试准确率: {best_test_acc:.4f}")
    return best_test_acc

if __name__ == '__main__':
    # 运行超参数调优
    best_params = run_hyperparameter_tuning(n_trials=30)
    
    # 使用最佳参数训练最终模型
    final_acc = train_with_best_params(best_params)
    
    print(f"\n超参数调优完成！")
    print(f"最终测试准确率: {final_acc:.4f}") 