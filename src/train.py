import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from models.biological_text_model import BiologicalTextModel
from data.race import RACEProcessor
from utils import accuracy
import config

class RACEDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        passage = sample['passage']
        question = sample['question']
        options = sample['options']
        label = ord(sample['answer']) - ord('A')
        x_top, x_base = self.processor.encode_sample(passage, question, options)
        return x_top.squeeze(0), x_base, label

def collate_fn(batch):
    # batch: list of (x_top, x_base, label)
    x_top = torch.stack([item[0] for item in batch], dim=0)  # [batch, 768]
    x_base = torch.stack([item[1] for item in batch], dim=0) # [batch, num_options, 768]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long) # [batch]
    return x_top, x_base, labels

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化数据处理器
    processor = RACEProcessor(config.DATA_DIR, config.BERT_MODEL_NAME, config.MAX_LENGTH)
    # 加载训练数据
    train_samples = processor.load_race('train')
    train_dataset = RACEDataset(train_samples, processor)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # 初始化模型
    model = BiologicalTextModel(config.INPUT_DIM_TOP, config.INPUT_DIM_BASE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # 训练循环
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0
        for x_top, x_base, label in train_loader:
            x_top = x_top.to(device)  # [batch, 768]
            x_base = x_base.to(device)  # [batch, num_options, 768]
            label = label.to(device)  # [batch]
            batch_size, num_options, hidden = x_base.size()
            # 扩展x_top到每个选项
            x_top_expand = x_top.unsqueeze(1).expand(-1, num_options, -1)  # [batch, num_options, 768]
            x_top_flat = x_top_expand.reshape(-1, hidden)  # [batch*num_options, 768]
            x_base_flat = x_base.reshape(-1, hidden)  # [batch*num_options, 768]
            # 模型输出
            logits = model(x_top_flat, x_base_flat).reshape(batch_size, num_options)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += accuracy(logits, label)
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={total_acc/len(train_loader):.4f}')
    # 保存模型
    torch.save(model.state_dict(), 'biological_race_model.pth') 