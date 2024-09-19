import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
import warnings
import logging
from datetime import datetime
import os
import jieba

warnings.filterwarnings('ignore')

# 设置日志
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_log_other_{current_time}.txt")
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_file}")
    return logger

# 设置日志目录
log_dir = "./logs/"
logger = setup_logger(log_dir)

def load_data(file_path, score_type='average'):
    logger.info(f"Loading data from {file_path} with score type: {score_type}")
    df = pd.read_csv(file_path)
    texts = df['message'].tolist()
    
    if score_type == 'score1':
        scores = df['score'].tolist()
    elif score_type == 'score2':
        scores = df['socre2'].tolist()  # 注意：这里使用了原始数据中的拼写 'socre2'
    elif score_type == 'average':
        scores = df[['score', 'socre2']].mean(axis=1).tolist()
    else:
        raise ValueError("Invalid score_type. Choose 'score1', 'score2', or 'average'.")
    
    logger.info(f"Loaded {len(texts)} samples")
    return texts, scores

# 模型定义
class FastTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(FastTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

class TextCNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes):
        super(TextCNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (k, embedding_dim)) for k in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers):
        super(TransformerEncoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

def get_model(model_name, vocab_size):
    if model_name == 'fasttext':
        return FastTextModel(vocab_size, 100)
    elif model_name == 'textcnn':
        return TextCNNModel(vocab_size, 100, 100, [3, 4, 5])
    elif model_name == 'lstm':
        return LSTMModel(vocab_size, 100, 128)
    elif model_name == 'transformer':
        return TransformerEncoderModel(vocab_size, 100, 4, 2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def preprocess_text(texts):
    return [list(jieba.cut(text)) for text in texts]

def build_vocab(tokenized_texts):
    word_to_ix = {}
    for sent in tokenized_texts:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    word_to_ix['<PAD>'] = len(word_to_ix)
    word_to_ix['<UNK>'] = len(word_to_ix)
    return word_to_ix

class TextScoreDataset(Dataset):
    def __init__(self, texts, scores, word_to_ix, max_length):
        self.texts = texts
        self.scores = scores
        self.word_to_ix = word_to_ix
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        score = self.scores[idx]
        indexed = [self.word_to_ix.get(word, self.word_to_ix['<UNK>']) for word in text]
        if len(indexed) < self.max_length:
            indexed += [self.word_to_ix['<PAD>']] * (self.max_length - len(indexed))
        else:
            indexed = indexed[:self.max_length]
        return {
            'input_ids': torch.tensor(indexed, dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.float)
        }

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        scores = batch['score'].to(device)
        outputs = model(input_ids).squeeze()
        loss = criterion(outputs, scores)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            scores = batch['score'].to(device)
            outputs = model(input_ids).squeeze()
            predictions.extend(outputs.tolist())
            actual.extend(scores.tolist())
    return np.mean(np.abs(np.array(predictions) - np.array(actual)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(model_name, X_train, X_test, y_train, y_test, score_type, model_dir):
    logger.info(f"Starting experiment for {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    tokenized_texts = preprocess_text(X_train + X_test)
    word_to_ix = build_vocab(tokenized_texts)
    vocab_size = len(word_to_ix)
    model = get_model(model_name, vocab_size)
    
    # 计算并记录参数量
    param_count = count_parameters(model)
    logger.info(f"{model_name.upper()} - Total trainable parameters: {param_count:,}")
    
    model.to(device)

    max_length = 256
    batch_size = 32
    train_dataset = TextScoreDataset([list(jieba.cut(text)) for text in X_train], y_train, word_to_ix, max_length)
    test_dataset = TextScoreDataset([list(jieba.cut(text)) for text in X_test], y_test, word_to_ix, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    logger.info(f"Training {model_name} for {num_epochs} epochs")
    start_time = time.time()
    best_mae = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        test_mae = evaluate(model, test_loader, device)
        logger.info(f"{model_name.upper()} - Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test MAE: {test_mae:.4f}")
        
        if test_mae < best_mae:
            best_mae = test_mae
            logger.info(f"New best MAE: {best_mae:.4f}")
            model_save_path = os.path.join(model_dir, f"best_{model_name}_{score_type}_model.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model to {model_save_path}")
    
    total_time = time.time() - start_time
    logger.info(f"Experiment for {model_name} completed. Best MAE: {best_mae:.4f}, Total time: {total_time:.2f} seconds")
    return best_mae, total_time

def main():
    start_time = time.time()
    logger.info("Starting NLP model comparison experiment")
    
    file_path = '最终标注结果4000.csv'  # 请替换为您的 CSV 文件路径
    score_types = ['score1', 'score2']
    model_names = ['fasttext', 'textcnn', 'lstm', 'transformer']

    # 创建保存模型的目录
    model_dir = "./models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for score_type in score_types:
        logger.info(f"\n\nStarting experiments using {score_type}")
        logger.info("============================")
        
        texts, scores = load_data(file_path, score_type)
        X_train, X_test, y_train, y_test = train_test_split(texts, scores, test_size=0.2, random_state=42)
        logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
        
        results = {}
        for model_name in model_names:
            logger.info(f"\nRunning experiment for {model_name.upper()} with {score_type}")
            mae, training_time = run_experiment(model_name, X_train, X_test, y_train, y_test, score_type, model_dir)
            results[model_name] = {'MAE': mae, 'Training Time': training_time}

        # 打印结果比较
        logger.info(f"\nFinal Results for {score_type}:")
        logger.info("--------------")
        for model_name, result in results.items():
            logger.info(f"{model_name.upper()}:")
            logger.info(f"  MAE: {result['MAE']:.4f}")
            logger.info(f"  Training Time: {result['Training Time']:.2f} seconds")
        logger.info("--------------")

        # 找出性能最好的模型
        best_model = min(results, key=lambda x: results[x]['MAE'])
        logger.info(f"Best performing model for {score_type}: {best_model.upper()} with MAE of {results[best_model]['MAE']:.4f}")

    total_experiment_time = time.time() - start_time
    logger.info(f"\nTotal experiment time: {total_experiment_time:.2f} seconds")

if __name__ == "__main__":
    main()