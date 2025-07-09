import torch

def accuracy(preds, labels):
    """
    计算多选题准确率
    preds: [batch_size, num_options]，每个选项的分数
    labels: [batch_size]，正确选项的索引
    """
    pred_choice = torch.argmax(preds, dim=1)
    correct = (pred_choice == labels).sum().item()
    return correct / len(labels)

# 可扩展：批处理、日志等 