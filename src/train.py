import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.biological_text_model import BiologicalTextModel
from data.race import RACEProcessor
from utils import accuracy
import config

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    # 初始化数据处理器
    processor = RACEProcessor(config.DATA_DIR, config.BERT_MODEL_NAME, config.MAX_LENGTH)
    # 加载训练数据
    train_samples = processor.load_race('train')
    # 先做特征提取与分块缓存
    processor.extract_and_cache_features(train_samples, 'train', chunk_size=1000)
    # 分块训练
    model = BiologicalTextModel(config.INPUT_DIM_TOP, config.INPUT_DIM_BASE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0
        total_batches = 0
        for x_top, x_base, labels in processor.load_cached_features_chunked('train'):
            train_dataset = TensorDataset(x_top, x_base, labels)
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            for x_top_b, x_base_b, label_b in train_loader:
                x_top_b = x_top_b.to(device)  # [batch, 768]
                x_base_b = x_base_b.to(device)  # [batch, num_options, 768]
                label_b = label_b.to(device)  # [batch]
                batch_size, num_options, hidden = x_base_b.size()
                # 扩展x_top到每个选项
                x_top_expand = x_top_b.unsqueeze(1).expand(-1, num_options, -1)  # [batch, num_options, 768]
                x_top_flat = x_top_expand.reshape(-1, hidden)  # [batch*num_options, 768]
                x_base_flat = x_base_b.reshape(-1, hidden)  # [batch*num_options, 768]
                # 模型输出
                logits = model(x_top_flat, x_base_flat).reshape(batch_size, num_options)
                loss = criterion(logits, label_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_acc += accuracy(logits, label_b)
                total_batches += 1
        print(f'Epoch {epoch+1}: Loss={total_loss/total_batches:.4f}, Acc={total_acc/total_batches:.4f}')
    # 保存模型
    torch.save(model.state_dict(), 'biological_race_model.pth') 