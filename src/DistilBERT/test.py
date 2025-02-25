from models.DistilBERT import BertClassify
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='alpaca_data_cleaned.json')
    parser.add_argument("--train_data_path", type=str, default='train_dataset.json')
    parser.add_argument("--test_data_path", type=str, default='test_dataset.json')
    parser.add_argument("--n_classes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default='1e-5')
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    return args

# Encode texts using DistilBERT tokenizer
def encode_texts(texts):
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Count samples within specified intervals
def count_samples_in_intervals(predictions, intervals):
    counts = {f"{interval[0]:.1f}-{interval[1]:.1f}": 0 for interval in intervals}
    for prediction in predictions:
        for interval in intervals:
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

if __name__ == '__main__':
    test()
