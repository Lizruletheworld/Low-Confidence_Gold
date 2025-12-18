from models.DistilBERT import BertClassify
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import argparse
import json
import re
import os

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='alpaca_data_cleaned.json')
    parser.add_argument("--train_data_path", type=str, default='train_dataset.json')
    parser.add_argument("--test_data_path", type=str, default='alpaca_labels.json')
    parser.add_argument("--n_classes", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=str, default='1e-5') # 修改为str以防报错
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_size", type=int, default=5200)
    args = parser.parse_args()
    return args

# 自定义一个健壮的 JSON 加载函数，解决 Trailing Data 问题
def safe_load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式匹配所有的 { ... } 块
    # re.DOTALL 确保 . 可以匹配换行符
    # \{ 和 \} 匹配字面量大括号
    # .*? 是非贪婪匹配，确保匹配到最近的结束括号
    json_objects = re.findall(r'\{.*?\}', content, re.DOTALL)
    
    data_list = []
    for obj_str in json_objects:
        try:
            # 尝试解析匹配到的每一个块
            data_list.append(json.loads(obj_str))
        except json.JSONDecodeError:
            # 如果解析失败，可能是匹配到了嵌套括号或其他干扰，跳过
            continue

    if not data_list:
        raise ValueError(f"无法从文件 {file_path} 中解析出任何有效的 JSON 对象")
        
    return pd.DataFrame(data_list)

# Encode texts using DistilBERT tokenizer
def encode_texts(texts):
    # 使用本地缓存或镜像下载
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Data preprocessing function
def data_process():
    args = parse_args()

    # --- 修改点 1: 使用更稳妥的加载方式 ---
    new_dataframe = safe_load_json(args.test_data_path)
    
    # 确保列名存在，防止数字或None导致拼接失败
    texts = [str(inst) + ' ' + str(inp) for inst, inp in zip(new_dataframe['instruction'], new_dataframe['input'])]
    
    encoded_inputs = encode_texts(texts)
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader, new_dataframe # 返回 df 避免二次读取

# Test function
def test():
    args = parse_args()
    # --- 修改点 2: 统一数据流 ---
    dataloader, new_dataframe = data_process()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassify(args.n_classes).to(device)
    
    model_path = 'distilbert_state_dict.pth'
    if not os.path.exists(model_path):
        print(f"Error: 找不到模型权重文件 {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    print("开始推理...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch, attention_mask_batch = batch
            outputs = model(input_ids_batch.to(device), attention_mask_batch.to(device))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions.extend(probabilities.cpu().numpy())

    predicted_classes = [np.argmax(p) for p in predictions]
    targets = new_dataframe['cluster'].values
    
    accuracy = np.mean(np.array(predicted_classes) == targets)
    print(f'Accuracy: {accuracy:.4f}')

    # 找出预测错误的索引
    incorrect_predictions_idx = np.where(np.array(predicted_classes) != targets)[0]
    incorrect_confidences = [predictions[i][predicted_classes[i]] for i in incorrect_predictions_idx]
    
    sorted_incorrect = sorted(zip(incorrect_predictions_idx, incorrect_confidences), key=lambda x: x[1])

    # 提取低置信度样本
    top_incorrect_idx = [idx for idx, _ in sorted_incorrect[:args.sample_size]]
    
    # --- 修改点 3: 避免 load_dataset 可能引发的路径问题，直接从 dataframe 取数据 ---
    low_confidence_full_data = new_dataframe.iloc[top_incorrect_idx].to_dict(orient='records')

    filename = 'distilbert_low_confidence.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(low_confidence_full_data, f, indent=4, ensure_ascii=False)
    
    print(f"已保存低置信度数据至: {filename}")

if __name__ == '__main__':
    test()
            if interval[0] <= prediction < interval[1]:
                counts[f"{interval[0]:.1f}-{interval[1]:.1f}"] += 1
                break
    return counts

# Data preprocessing function
def data_process():
    args = parse_args()

    new_dataframe = pd.read_json(args.test_data_path)
    X_labeled = new_dataframe.apply(lambda row: {'input': row['input'], 'instruction': row['instruction']},
                                    axis=1).tolist()
    texts = [x['input'] + ' ' + x['instruction'] for x in X_labeled]
    encoded_inputs = encode_texts(texts)
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    # Create TensorDataset
    dataset = TensorDataset(input_ids, attention_mask)

    # Create DataLoader
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Test function
def test():
    dataloader = data_process()
    args = parse_args()
    new_dataframe = pd.read_json(args.test_data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU or CPU
    model = BertClassify(args.n_classes).to(device)
    model_path = 'distilbert_state_dict.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch, attention_mask_batch = batch
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)

            # Forward pass
            outputs = model(input_ids_batch, attention_mask_batch)
            # Get prediction probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions.extend(probabilities.cpu().numpy())

    # Get predicted classes
    predicted_classes = [np.argmax(prediction) for prediction in predictions]
    targets = new_dataframe['cluster'].values
    # Calculate accuracy
    accuracy = np.mean(np.array(predicted_classes) == targets)
    print(f'Accuracy: {accuracy}')

    incorrect_predictions_idx  = np.where(np.array(predicted_classes) != targets)[0]

    # Extract the confidence of samples with prediction errors.
    incorrect_confidences = [predictions[i][predicted_classes[i]] for i in incorrect_predictions_idx]

    # Combine the indices and confidence levels of the incorrectly predicted samples, and sort them by confidence level.
    sorted_incorrect_by_confidence = sorted(zip(incorrect_predictions_idx, incorrect_confidences), key=lambda x: x[1])


    # Get top indices with lowest confidence
    top_incorrect_idx = [idx for idx, _ in sorted_incorrect_by_confidence[:args.sample_size]]

    # Extract complete records for these indices from the original dataset
    file_path = args.test_data_path
    ds = load_dataset('json', data_files=file_path)
    low_confidence_full_data = [ds['train'][int(i)] for i in top_incorrect_idx]

    # Save to JSON file
    filename = f'distilbert_low_confidence.json'
    with open(filename, 'w') as f:
        json.dump(low_confidence_full_data, f)





if __name__ == '__main__':
    test()
