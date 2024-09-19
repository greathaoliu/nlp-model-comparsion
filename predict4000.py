import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import warnings
import os
import math
import csv
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_CACHE_DIR"] = "/mnt/data21/liuhao/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings('ignore')

# Load model and tokenizer 
def load_model_and_tokenizer(model_name, model_path):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1) 
        model.load_state_dict(torch.load(model_path)) 
        model.eval()     
        tokenizer = AutoTokenizer.from_pretrained(model_name) 
        return model, tokenizer

# Updated load_prediction_data function
def load_prediction_data(file_path):
    def read_csv_robust(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Use csv.reader with the appropriate quoting settings
            reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            headers = next(reader)
            data = []
            for i, row in enumerate(reader, start=2):
                try:
                    if len(row) == len(headers):
                        data.append(row)
                    else:
                        print(f"Skipping malformed row {i}: {row}")
                except Exception as e:
                    print(f"Error reading row {i}: {e}")
        return pd.DataFrame(data, columns=headers)

    try:
        # Try reading with pandas first
        df = pd.read_csv(file_path, quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True, escapechar='\\')
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        print("Attempting to read file with custom method...")
        df = read_csv_robust(file_path)
    
    return df
    
# Updated prediction function
def predict(model, tokenizer, texts, max_length=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions = []
    with torch.no_grad():
        for text in tqdm(texts):
            if pd.isna(text) or not isinstance(text, str):
                predictions.append(0)
            else:
                encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                prediction = outputs.logits.squeeze().item()
                predictions.append(prediction if not math.isnan(prediction) else 0)
    
    return predictions

# Main prediction pipeline
def main():
    model = 'xlnet'

    model_name = ''
    model1_path = ''
    model2_path = ''
    
    if model == 'roberta':
        model_name = "hfl/chinese-roberta-wwm-ext-large"
        model1_path = "./models/final_roberta_score1_model.pth"
        model2_path = "./models/final_roberta_score2_model.pth"
    elif model == 'xlnet':
        model_name = "hfl/chinese-xlnet-mid"
        model1_path = "./models/final_xlnet_score1_model.pth"
        model2_path = "./models/final_xlnet_score2_model.pth"

    # Load the models and tokenizers
    model1, tokenizer1 = load_model_and_tokenizer(model_name, model1_path)
    model2, tokenizer2 = load_model_and_tokenizer(model_name, model2_path)

    # Load the prediction data
    file_path = f'最终标注结果4000.csv'
    df = load_prediction_data(file_path)
    
    # Extract the "留言内容" column for prediction
    texts = df['message'].tolist()
    
    # Perform predictions using RoBERTa for "办理态度评分" and "办理速度评分" (score1)
    print(f"Predicting score1 using {model}...")
    score1_predictions = predict(model1, tokenizer1, texts)
    
    # Perform predictions using XLNet for "解决程度评分" (score2)
    print(f"Predicting score2 using {model}...")
    score2_predictions = predict(model2, tokenizer2, texts)
    
    # Add predictions to the dataframe
    df['score1_预测'] = score1_predictions
    df['score2_预测'] = score2_predictions
    
    # Save the results to a new CSV
    df.to_csv(f'4000-{model}.csv', index=False)
    print(f"Predictions saved to '4000-{model}.csv'")

if __name__ == "__main__":
    main()