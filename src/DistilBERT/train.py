from models.DistilBERT import BertClassify
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer
import torch
import argparse
import os

# 使用全局设备变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='alpaca_data_cleaned.json')
    parser.add_argument("--train_data_path", type=str, default='train_dataset.json')
    parser.add_argument("--n_classes", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5) # 确保是 float
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    return args

def encode_texts(texts):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

def data_process():
    args = parse_args()
    # 建议使用变量名 args.train_data_path 而不是硬编码
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"找不到训练文件: {args.train_data_path}")
        
    new_dataframe = pd.read_json(args.train_data_path)
    X_labeled = new_dataframe.apply(lambda row: {'input': row['input'], 'instruction': row['instruction']}, axis=1).tolist()
    y_labeled = new_dataframe['cluster'].values
    
    combined_texts = [str(x['input']) + ' ' + str(x['instruction']) for x in X_labeled]
    encoded_inputs = encode_texts(combined_texts)
    
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    labels = torch.tensor(y_labeled, dtype=torch.long) # 确保标签是长整型
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def train():
    args = parse_args()
    dataloader = data_process()

    model = BertClassify(args.n_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    print(f"开始在设备 {device} 上训练...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, label = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            # 这里的调用不需要传 token_type_ids
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} 完成，平均 Loss: {avg_loss:.4f}")

    # --- 关键：保存模型 ---
    model_save_path = 'distilbert_state_dict.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")
    
    return model

if __name__ == '__main__':
    train()

