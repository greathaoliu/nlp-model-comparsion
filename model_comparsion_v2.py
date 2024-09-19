import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
import warnings
import logging
from datetime import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HF_CACHE_DIR"]="/mnt/data21/liuhao/hf_cache"
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
warnings.filterwarnings('ignore')

# 设置日志
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_log_v2_{current_time}.txt")
    
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

class TextScoreDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        score = self.scores[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, 
                                  padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': torch.tensor(score, dtype=torch.float)
        }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_and_tokenizer(model_name):
    model_configs = {
        'bert': 'bert-base-chinese',
        'roberta': 'hfl/chinese-roberta-wwm-ext',
        'xlnet': 'hfl/chinese-xlnet-base',
        'albert': 'voidful/albert_chinese_base',
        'electra': 'hfl/chinese-electra-base-discriminator',
        'ernie': 'nghuyong/ernie-3.0-base-zh',
        'deberta': 'IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese'
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model_path = model_configs[model_name]
    logger.info(f"Loading model and tokenizer for {model_name} from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 计算并记录参数量
    total_params = count_parameters(model)
    logger.info(f"{model_name.upper()} - Total trainable parameters: {total_params:,}")
    
    return model, tokenizer

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=scores)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.logits.squeeze().tolist())
            actual.extend(scores.tolist())
    return np.mean(np.abs(np.array(predictions) - np.array(actual)))

def run_experiment(model_name, X_train, X_test, y_train, y_test, score_type, model_dir):
    logger.info(f"Starting experiment for {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model, tokenizer = get_model_and_tokenizer(model_name)
    model.to(device)

    max_length = 256
    batch_size = 16
    train_dataset = TextScoreDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = TextScoreDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    num_epochs = 10
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    logger.info(f"Training {model_name} for {num_epochs} epochs")
    start_time = time.time()
    best_mae = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, scheduler, device)
        test_mae = evaluate(model, test_loader, device)
        logger.info(f"{model_name.upper()} - Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test MAE: {test_mae:.4f}")
        
        if test_mae < best_mae:
            best_mae = test_mae
            logger.info(f"New best MAE: {best_mae:.4f}")
            # 保存最佳模型
            model_save_path = os.path.join(model_dir, f"best_{model_name}_{score_type}_model.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model to {model_save_path}")
    
    total_time = time.time() - start_time
    logger.info(f"Experiment for {model_name} completed. Best MAE: {best_mae:.4f}, Total time: {total_time:.2f} seconds")
    return best_mae, total_time

# 主实验流程
def main():
    start_time = time.time()
    logger.info("Starting NLP model comparison experiment")
    
    file_path = '最终标注结果4000.csv'  # 请替换为您的 CSV 文件路径
    score_types = ['score1', 'score2']
    model_names = ['bert', 'roberta', 'xlnet', 'albert', 'electra', 'ernie', 'deberta']

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