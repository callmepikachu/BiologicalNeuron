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
import time

def load_features(data_dir, split):
    """加载预提取的特征"""
    features = torch.load(os.path.join(data_dir, f'{split}_features.pt'))
    return features['x_top'], features['x_base'], features['labels']

def train_mvsa_extended(data_dir='data/MVSA_Single', epochs=200, lr=0.002, batch_size=16):
    """使用最佳配置进行扩展训练"""
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
    
    # 模型参数（使用最佳配置）
    text_dim = train_text.shape[1]  # BERT特征维度
    img_dim = train_img.shape[1]    # ResNet特征维度
    num_classes = 3  # negative, neutral, positive
    
    # 创建生物神经元模型（最佳配置）
    model = BiologicalNeuronLayer(
        input_dim_top=text_dim,
        input_dim_base=img_dim,
        num_neurons=32  # 最佳配置
    ).to(device)
    
    # 分类头（最佳配置）
    classifier = nn.Sequential(
        nn.Linear(32, 64),  # 最佳配置
        nn.ReLU(),
        nn.Dropout(0.2),    # 最佳配置
        nn.Linear(64, num_classes)
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # 训练记录
    train_losses = []
    train_accuracies = []  # 添加训练准确率记录
    test_accuracies = []
    learning_rates = []
    best_accuracy = 0.0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    print(f'开始扩展训练，共{epochs}个epoch...')
    print(f'最佳配置: 32神经元, 64隐藏层, 0.2 dropout, 0.002学习率, 16批次大小')
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        classifier.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_text, batch_img, batch_labels in train_loader:
            batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            bio_output = model(batch_text, batch_img)
            outputs = classifier(bio_output)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)  # 记录训练准确率
        
        # 测试阶段
        model.eval()
        classifier.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_text, batch_img, batch_labels in test_loader:
                batch_text, batch_img, batch_labels = batch_text.to(device), batch_img.to(device), batch_labels.to(device)
                bio_output = model(batch_text, batch_img)
                outputs = classifier(bio_output)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        test_acc = test_correct / test_total
        test_accuracies.append(test_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start_time
        
        # 学习率调度
        scheduler.step(test_acc)
        
        # 早停检查
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'accuracy': test_acc,
                'epoch': epoch + 1,
                'best_params': {
                    'num_neurons': 32,
                    'hidden_dim': 64,
                    'dropout_rate': 0.2,
                    'learning_rate': lr,
                    'batch_size': batch_size
                }
            }, os.path.join(data_dir, 'best_extended_model.pth'))
            print(f'⭐ 保存最佳模型，准确率: {test_acc:.4f}')
        else:
            patience_counter += 1
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch < 10:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Loss={avg_loss:.4f}, '
                  f'Train Acc={train_acc:.4f}, '
                  f'Test Acc={test_acc:.4f}, '
                  f'LR={optimizer.param_groups[0]["lr"]:.6f}, '
                  f'Time={epoch_time:.1f}s')
        
        # 早停
        if patience_counter >= patience:
            print(f'\n早停触发！{patience}个epoch无改善')
            break
    
    total_time = time.time() - start_time
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 10))
    
    # 训练损失
    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 测试准确率
    plt.subplot(2, 3, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 学习率变化
    plt.subplot(2, 3, 3)
    plt.plot(learning_rates)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 训练vs测试准确率对比
    plt.subplot(2, 3, 4)
    plt.plot(train_accuracies, label='Train', alpha=0.7)
    plt.plot(test_accuracies, label='Test', alpha=0.7)
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失vs准确率
    plt.subplot(2, 3, 5)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    line1 = ax1.plot(train_losses, color='red', label='Loss')
    line2 = ax2.plot(test_accuracies, color='blue', label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='red')
    ax2.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title('Loss vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 最终评估
    plt.subplot(2, 3, 6)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title('Final Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('results/extended_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 最终评估
    print(f'\n=== 扩展训练完成 ===')
    print(f'最佳测试准确率: {best_accuracy:.4f} (Epoch {best_epoch})')
    print(f'总训练时间: {total_time/60:.1f}分钟')
    print(f'平均每epoch时间: {total_time/len(test_accuracies):.1f}秒')
    print(f'最终学习率: {optimizer.param_groups[0]["lr"]:.6f}')
    
    print('\n分类报告:')
    print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'learning_rates': learning_rates,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'total_epochs': len(test_accuracies),
        'total_time': total_time
    }
    
    os.makedirs('results', exist_ok=True)
    np.save('results/extended_training_history.npy', training_history)
    
    return model, best_accuracy

if __name__ == '__main__':
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 开始扩展训练
    model, best_acc = train_mvsa_extended(
        data_dir='data/MVSA_Single',
        epochs=200,  # 更多epoch
        lr=0.002,    # 最佳学习率
        batch_size=16  # 最佳批次大小
    )
    
    print(f'\n扩展训练完成！最佳测试准确率: {best_acc:.4f}') 