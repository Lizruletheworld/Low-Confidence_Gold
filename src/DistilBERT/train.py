from models.DistilBERT import BertClassify
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import argparse

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='alpaca_data_cleaned.json')
    parser.add_argument("--train_data_path", type=str, default='train_dataset.json')
    parser.add_argument("--n_classes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default='1e-5')
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    return args

def encode_texts(texts):
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

def data_process():
    args = parse_args()
    new_dataframe = pd.read_json('E:/alpaca/closest_texts_data_k_40_40.json')
    X_labeled = new_dataframe.apply(lambda row: {'input': row['input'], 'instruction': row['instruction']},
                                    axis=1).tolist()
    y_labeled = new_dataframe['cluster'].values
    combined_texts = [x['input'] + ' ' + x['instruction'] for x in X_labeled]
    encoded_inputs = encode_texts(combined_texts)
    # Create TensorDataset
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    labels = torch.tensor(y_labeled)
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def train():

    dataloader = data_process()

    args = parse_args()

    model = BertClassify(args.n_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    # Train the model
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, label = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs} finished with average loss: {avg_loss}")

    return model

if __name__ == '__main__':
    train()
