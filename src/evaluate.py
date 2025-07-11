import torch
from torch.utils.data import DataLoader, TensorDataset
from models.biological_text_model import BiologicalTextModel
from data.race import RACEProcessor
from utils import accuracy
import config

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = RACEProcessor(config.DATA_DIR, config.BERT_MODEL_NAME, config.MAX_LENGTH)
    # 加载验证集样本
    dev_samples = processor.load_race('dev')
    # 自动分块提取特征（如已存在则跳过）
    processor.extract_and_cache_features(dev_samples, 'dev', chunk_size=1000)
    # 加载模型
    model = BiologicalTextModel(config.INPUT_DIM_TOP, config.INPUT_DIM_BASE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    model.load_state_dict(torch.load('biological_race_model.pth', map_location=device))
    model.to(device)
    model.eval()
    total_acc = 0
    total_batches = 0
    with torch.no_grad():
        for x_top, x_base, labels in processor.load_cached_features_chunked('dev'):
            dev_dataset = TensorDataset(x_top, x_base, labels)
            dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            for x_top_b, x_base_b, label_b in dev_loader:
                x_top_b = x_top_b.to(device)
                x_base_b = x_base_b.to(device)
                label_b = label_b.to(device)
                batch_size, num_options, hidden = x_base_b.size()
                x_top_expand = x_top_b.unsqueeze(1).expand(-1, num_options, -1)
                x_top_flat = x_top_expand.reshape(-1, hidden)
                x_base_flat = x_base_b.reshape(-1, hidden)
                logits = model(x_top_flat, x_base_flat).reshape(batch_size, num_options)
                total_acc += accuracy(logits, label_b)
                total_batches += 1
    print(f'验证集准确率: {total_acc/total_batches:.4f}') 