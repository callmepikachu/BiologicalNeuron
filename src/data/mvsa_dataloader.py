import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer

def robust_read_text(path):
    for encoding in ['utf-8', 'gbk', 'latin1']:
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"无法解码文件: {path}")

class MVSASingleDataset(Dataset):
    def __init__(self, data_dir, split='train', max_length=128, transform=None):
        """
        data_dir: mvsa_single目录
        split: 'train' or 'test'
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # 读取标签文件
        csv_path = os.path.join(data_dir, f'{split}.csv')
        self.data = pd.read_csv(csv_path)
        # 图片文件夹
        self.img_dir = os.path.join(data_dir, 'images')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 读取文本内容（自动兼容多种编码）
        text = robust_read_text(row['txt_path'])
        img_name = row['img_path']
        # 标签文本转数字
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        label_str = str(row['label']).strip().lower()
        label = label_map[label_str]
        # 文本embedding
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # 保证每个样本的input_ids/attention_mask shape为[seq_len]
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        # 图片
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, image, label 