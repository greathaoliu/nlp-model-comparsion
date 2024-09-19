import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
import pandas as pd
import warnings
import logging
from datetime import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_CACHE_DIR"] = "/mnt/data21/liuhao/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings('ignore')

# Setup logger
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_log_train_large_{current_time}.txt")
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_file}")
    return logger

log_dir = "./logs/"
logger = setup_logger(log_dir)

# Load data
def load_data(file_path, score_type='score1'):
    logger.info(f"Loading data from {file_path} with score type: {score_type}")
    df = pd.read_csv(file_path)
    texts = df['message'].tolist()
    
    if score_type == 'score1':
        scores = df['score'].tolist()
    elif score_type == 'score2':
        scores = df['socre2'].tolist()  # 注意拼写
    else:
        raise ValueError("Invalid score_type. Choose 'score1' or 'score2'.")
    
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
        'roberta': 'hfl/chinese-roberta-wwm-ext-large',
        'xlnet': 'hfl/chinese-xlnet-mid'
    }

    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model_path = model_configs[model_name]
    logger.info(f"Loading model and tokenizer for {model_name} from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
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

def run_experiment(model_name, texts, scores, score_type, model_dir):
    logger.info(f"Starting experiment for {model_name} on {score_type}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model, tokenizer = get_model_and_tokenizer(model_name)
    model.to(device)

    max_length = 256
    batch_size = 16
    dataset = TextScoreDataset(texts, scores, tokenizer, max_length)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    num_epochs = 10
    num_training_steps = num_epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    logger.info(f"Training {model_name} for {num_epochs} epochs")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, optimizer, scheduler, device)
        logger.info(f"{model_name.upper()} - Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
    
    # Save final model after all epochs for each score type
    model_save_path = os.path.join(model_dir, f"final_{model_name}_{score_type}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Saved final model to {model_save_path}")
    
    total_time = time.time() - start_time
    logger.info(f"Experiment for {model_name} on {score_type} completed. Total time: {total_time:.2f} seconds")
    return total_time

# Main experiment flow
def main():
    start_time = time.time()
    logger.info("Starting NLP model training experiment")
    
    file_path = '最终标注结果4000.csv'  # Replace with your CSV file path
    score_types = ['score1', 'score2']  # Train on both score1 and score2
    
    model_names = ['roberta', 'xlnet']  # Training for XLNet and RoBERTa

    # Create model directory
    model_dir = "./models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for score_type in score_types:
        texts, scores = load_data(file_path, score_type)
        logger.info(f"Data loaded for {score_type}: {len(texts)} samples")
        
        for model_name in model_names:
            logger.info(f"\nRunning experiment for {model_name.upper()} with {score_type}")
            training_time = run_experiment(model_name, texts, scores, score_type, model_dir)
            logger.info(f"{model_name.upper()} on {score_type} training completed in {training_time:.2f} seconds")

    total_experiment_time = time.time() - start_time
    logger.info(f"\nTotal experiment time: {total_experiment_time:.2f} seconds")

if __name__ == "__main__":
    main()
