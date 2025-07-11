import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
from torchvision import transforms, models
from data.mvsa_dataloader import MVSASingleDataset
from torch.utils.data import DataLoader
from transformers import BertModel

# ===== 自动生成MVSA-Single数据集索引csv（仅首次运行需要） =====
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_mvsa_csv(data_root):
    csv_files = [os.path.join(data_root, f) for f in ['train.csv', 'val.csv', 'test.csv']]
    if all(os.path.exists(f) for f in csv_files):
        print('train/val/test.csv 已存在，无需生成')
        return
    label_file = os.path.join(data_root, 'labelResultAll.txt')
    img_dir = os.path.join(data_root, 'data')
    # 读取标签（适配表头和编号）
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 跳过表头
        for line in lines[1:]:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 2:
                img_id, label = parts[0], parts[1]
                labels[f"{img_id}.jpg"] = label
    # 构建样本列表
    samples = []
    jpg_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f"图片文件数: {len(jpg_files)}")
    print(f"标签数: {len(labels)}")
    print("样例图片文件名：", jpg_files[:10])
    print("样例标签文件名：", list(labels.keys())[:10])
    for fname in jpg_files:
        img_path = os.path.join(img_dir, fname)
        txt_path = os.path.join(img_dir, fname.replace('.jpg', '.txt'))
        label = labels.get(fname, None)
        # 自动补零适配图片名
        if label is None:
            # 去掉扩展名后补零再加回扩展名
            name, ext = os.path.splitext(fname)
            for pad in range(1, 7):  # 最多补到6位
                fname_padded = name.zfill(len(name)+pad) + ext
                label = labels.get(fname_padded, None)
                if label is not None:
                    break
        if label is not None and os.path.exists(txt_path):
            samples.append([img_path, txt_path, label])
        else:
            if label is None:
                # print(f"标签缺失: {fname}")
                pass
            if not os.path.exists(txt_path):
                print(f"文本文件缺失: {txt_path}")
                pass
    print(f"有效样本数: {len(samples)}")
    if len(samples) == 0:
        print('未找到任何有效样本，请检查图片、文本和标签文件是否匹配！')
        return
    # 划分数据集
    train, test = train_test_split(samples, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    # 保存为csv
    pd.DataFrame(train, columns=['img_path', 'txt_path', 'label']).to_csv(os.path.join(data_root, 'train.csv'), index=False)
    pd.DataFrame(val, columns=['img_path', 'txt_path', 'label']).to_csv(os.path.join(data_root, 'val.csv'), index=False)
    pd.DataFrame(test, columns=['img_path', 'txt_path', 'label']).to_csv(os.path.join(data_root, 'test.csv'), index=False)
    print('已自动生成 train.csv, val.csv, test.csv')


def extract_features(data_dir, split='train', batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 文本BERT
    bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert.eval()
    # 图像ResNet18
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()
    # 数据集
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = MVSASingleDataset(data_dir, split=split, transform=img_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_text_emb, all_img_emb, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            text_inputs, images, labels = batch
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            text_emb = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            images = images.to(device)
            img_emb = resnet(images)
            all_text_emb.append(text_emb.cpu())
            all_img_emb.append(img_emb.cpu())
            all_labels.append(labels)
    torch.save({
        'x_top': torch.cat(all_text_emb),
        'x_base': torch.cat(all_img_emb),
        'labels': torch.cat(all_labels)
    }, os.path.join(data_dir, f'{split}_features.pt'))
    print(f'{split}特征已保存')

if __name__ == '__main__':
    generate_mvsa_csv('data/MVSA_Single')
    extract_features('data/MVSA_Single', split='train')
    extract_features('data/MVSA_Single', split='test') 