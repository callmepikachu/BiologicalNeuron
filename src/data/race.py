import os
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch

class RACEProcessor:
    def __init__(self, data_dir, bert_model_name='bert-base-uncased', max_length=512):
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.max_length = max_length
        self.bert.eval()

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
                            'answer': answer
                        })
        return data

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