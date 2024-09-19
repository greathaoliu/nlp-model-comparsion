import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
import warnings
import logging
import os
from datetime import datetime
from gensim.models import Word2Vec
import jieba

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HF_CACHE_DIR"]="/mnt/data21/liuhao/hf_cache"
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
warnings.filterwarnings('ignore')

# 设置日志
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_log_word2vec_{current_time}.txt")
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_file}")
    return logger

# 设置日志和模型保存目录
base_dir = "."  # 请替换为您想要保存日志和模型的基础目录
log_dir = os.path.join(base_dir, "logs")
model_dir = os.path.join(base_dir, "models")
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

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Word2VecModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        hidden = self.relu(self.fc1(pooled))
        output = self.fc2(hidden)
        return output

class Word2VecDataset(Dataset):
    def __init__(self, texts, scores, word2idx, max_length):
        self.texts = texts
        self.scores = scores
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        score = self.scores[idx]
        words = jieba.lcut(text)
        indexed = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        if len(indexed) < self.max_length:
            indexed += [self.word2idx['<PAD>']] * (self.max_length - len(indexed))
        else:
            indexed = indexed[:self.max_length]
        return {
            'input_ids': torch.tensor(indexed, dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.float)
        }

def train_word2vec(texts):
    logger.info("Training Word2Vec model")
    segmented_texts = [jieba.lcut(text) for text in texts]
    model = Word2Vec(sentences=segmented_texts, vector_size=100, window=5, min_count=1, workers=4)
    logger.info("Word2Vec model training completed")
    return model

def get_word2vec_model(texts):
    word2vec_model = train_word2vec(texts)
    vocab = list(word2vec_model.wv.key_to_index.keys())
    word2idx = {word: i+2 for i, word in enumerate(vocab)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    
    embedding_matrix = np.zeros((len(word2idx), 100))
    for word, i in word2idx.items():
        if i >= 2:
            embedding_matrix[i] = word2vec_model.wv[word]
    
    model = Word2VecModel(len(word2idx), 100, 64)
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    return model, word2idx

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def run_word2vec_experiment(X_train, X_test, y_train, y_test, score_type):
    logger.info("Starting experiment for Word2Vec model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model, word2idx = get_word2vec_model(X_train + X_test)
        # 计算并记录参数量
    total_params = count_parameters(model)
    logger.info(f"Word2Vec - Total trainable parameters: {total_params:,}")

    model.to(device)

    max_length = 256
    batch_size = 32
    train_dataset = Word2VecDataset(X_train, y_train, word2idx, max_length)
    test_dataset = Word2VecDataset(X_test, y_test, word2idx, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 10
    logger.info(f"Training Word2Vec model for {num_epochs} epochs")
    start_time = time.time()
    best_mae = float('inf')

    model_save_dir = os.path.join(model_dir, score_type)
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            scores = batch['score'].to(device)
            outputs = model(input_ids).squeeze()
            loss = criterion(outputs, scores)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        predictions = []
        actual = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                scores = batch['score'].to(device)
                outputs = model(input_ids).squeeze()
                predictions.extend(outputs.tolist())
                actual.extend(scores.tolist())

        test_mae = np.mean(np.abs(np.array(predictions) - np.array(actual)))
        logger.info(f"Word2Vec - Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Test MAE: {test_mae:.4f}")

        if test_mae < best_mae:
            best_mae = test_mae
            logger.info(f"New best MAE: {best_mae:.4f}")
            best_model_path = os.path.join(model_save_dir, "best_word2vec_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model to {best_model_path}")

    final_model_path = os.path.join(model_save_dir, "final_word2vec_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    total_time = time.time() - start_time
    logger.info(f"Experiment for Word2Vec completed. Best MAE: {best_mae:.4f}, Total time: {total_time:.2f} seconds")
    return best_mae, total_time

if __name__ == "__main__":
    logger.info("Starting Word2Vec text scoring experiment")
    file_path = './最终标注结果4000.csv'  # 请替换为您的 CSV 文件路径
    score_types = ['score1', 'score2']

    for score_type in score_types:
        logger.info(f"\n\nExperiments using {score_type}")
        logger.info("============================")
        
        texts, scores = load_data(file_path, score_type)
        X_train, X_test, y_train, y_test = train_test_split(texts, scores, test_size=0.2, random_state=42)
        logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
        
        mae, training_time = run_word2vec_experiment(X_train, X_test, y_train, y_test, score_type)
        
        logger.info("\nFinal Results:")
        logger.info("--------------")
        logger.info(f"Score Type: {score_type}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"Training Time: {training_time:.2f} seconds")
        logger.info("--------------")

    logger.info("Word2Vec text scoring experiment completed")