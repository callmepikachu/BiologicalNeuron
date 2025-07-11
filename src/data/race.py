import os
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch

class RACEProcessor:
    def __init__(self, data_dir, bert_model_name='bert-base-uncased', max_length=512, cache_dir=None):
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.max_length = max_length
        self.bert.eval()
        self.cache_dir = cache_dir or os.path.join(data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_race(self, split='train'):
        """
        加载RACE数据集，split为'train'/'dev'/'test'
        返回：样本列表，每个样本为dict（每个问题为一个样本）
        """
        data = []
        for level in ['high', 'middle']:
            folder = os.path.join(self.data_dir, split, level)
            if not os.path.exists(folder):
                continue
            for fname in tqdm(os.listdir(folder), desc=f'加载{split}-{level}'):
                if not fname.endswith('.txt'):
                    continue
                with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                    sample = json.load(f)
                    passage = sample['article']
                    for i, question in enumerate(sample['questions']):
                        options = sample['options'][i]
                        answer = sample['answers'][i]
                        data.append({
                            'passage': passage,
                            'question': question,
                            'options': options,
                            'answer': answer,
                            'split': split,
                            'level': level,
                            'fname': fname,
                            'q_idx': i
                        })
        return data

    def extract_and_cache_features(self, samples, split, chunk_size=1000):
        """
        分块提取BERT特征，每chunk_size条保存一个pt文件
        """
        total = len(samples)
        num_chunks = (total + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            cache_path = os.path.join(self.cache_dir, f'{split}_features_part{chunk_idx}.pt')
            if os.path.exists(cache_path):
                print(f'已存在缓存: {cache_path}')
                continue
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, total)
            batch = samples[start:end]
            batch_x_top, batch_x_base, batch_labels = [], [], []
            for sample in tqdm(batch, desc=f'提取{split}特征[{chunk_idx+1}/{num_chunks}]'):
                passage = sample['passage']
                question = sample['question']
                options = sample['options']
                label = ord(sample['answer']) - ord('A')
                # 段落级特征
                inputs = self.tokenizer(passage, truncation=True, max_length=self.max_length, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.bert(**inputs)
                x_top = outputs.last_hidden_state[:, 0, :].squeeze(0)  # [768]
                # 选项特征
                x_base = []
                for opt in options:
                    q_opt = question + ' ' + opt
                    inputs = self.tokenizer(q_opt, truncation=True, max_length=self.max_length, return_tensors='pt')
                    with torch.no_grad():
                        outputs = self.bert(**inputs)
                    x_base.append(outputs.last_hidden_state.mean(dim=1).squeeze(0))
                x_base = torch.stack(x_base, dim=0)  # [num_options, 768]
                batch_x_top.append(x_top)
                batch_x_base.append(x_base)
                batch_labels.append(label)
            torch.save({'x_top': torch.stack(batch_x_top), 'x_base': torch.stack(batch_x_base), 'labels': torch.tensor(batch_labels)}, cache_path)
            print(f'特征已缓存到: {cache_path}')

    def get_feature_chunk_paths(self, split):
        """
        获取所有分块缓存文件路径
        """
        files = []
        for fname in os.listdir(self.cache_dir):
            if fname.startswith(f'{split}_features_part') and fname.endswith('.pt'):
                files.append(os.path.join(self.cache_dir, fname))
        files.sort()  # 保证顺序
        return files

    def load_cached_features_chunked(self, split):
        """
        生成器，每次yield一个分块的(x_top, x_base, labels)
        """
        chunk_paths = self.get_feature_chunk_paths(split)
        for path in chunk_paths:
            data = torch.load(path)
            yield data['x_top'], data['x_base'], data['labels']

    def encode_sample(self, passage, question, options):
        """
        对单个样本进行BERT编码，返回段落特征（x_top）、选项特征（x_base）
        """
        # 段落级特征（[CLS]）
        inputs = self.tokenizer(passage, truncation=True, max_length=self.max_length, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert(**inputs)
        x_top = outputs.last_hidden_state[:, 0, :]  # [CLS]向量
        # 选项级特征（每个选项+问题）
        x_base = []
        for opt in options:
            q_opt = question + ' ' + opt
            inputs = self.tokenizer(q_opt, truncation=True, max_length=self.max_length, return_tensors='pt')
            with torch.no_grad():
                outputs = self.bert(**inputs)
            x_base.append(outputs.last_hidden_state.mean(dim=1))  # 取平均池化
        x_base = torch.cat(x_base, dim=0)  # [num_options, hidden_size]
        return x_top, x_base 