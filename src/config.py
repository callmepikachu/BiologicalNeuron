# 配置文件，集中管理超参数和路径

# 数据路径
DATA_DIR = './data/RACE'

# 预训练BERT模型名
BERT_MODEL_NAME = 'bert-base-uncased'

# 最大输入长度
MAX_LENGTH = 512

# 神经元层参数
INPUT_DIM_TOP = 768  # BERT输出维度
INPUT_DIM_BASE = 768
HIDDEN_SIZE = 32
OUTPUT_SIZE = 4  # RACE为4选项

# 训练参数
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5 