from transformers import DistilBertTokenizer, DistilBertModel
import argparse
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='alpaca_data_cleaned.json')
    parser.add_argument("--train_data_path", type=str, default='train_dataset.json')
    parser.add_argument("--test_data_path", type=str, default='alpaca_labels.json')
    parser.add_argument("--n_classes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default='1e-5')
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_size", type=int, default=5200)
    args = parser.parse_args()
    return args

# Encode texts
def encode_texts(texts):
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

def data_process():
    args = parse_args()

    # Specify the path to the dataset
    file_path = args.test_data_path

    # Call the function and receive two objects in return
    new_dataframe = pd.read_json(args.train_data_path)

    # Separate out the labeled data
    X_labeled = new_dataframe.apply(lambda row: {'input': row['input'], 'instruction': row['instruction']},
                                    axis=1).tolist()

    ds = load_dataset('json', data_files=file_path)
    texts = [input_text + ' ' + instruction_text for input_text, instruction_text in
             zip(ds['train']['input'], ds['train']['instruction'])]
    answer = ds['train']['cluster']

    train_texts = [x['input'] + ' ' + x['instruction'] for x in X_labeled]

    return train_texts, texts, answer

def multinomialnb():
    args = parse_args()
    train_texts, texts, answer = data_process()
    # Create a workflow pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB()),
    ])

    # Train the model using the labeled dataset
    pipeline.fit(train_texts, answer)

    y_pred = pipeline.predict(texts)

    # Calculate accuracy
    accuracy = accuracy_score(answer, y_pred)
    print(f"Accuracy: {accuracy}")

    # Use the model to predict probabilities for unlabeled data
    proba = pipeline.predict_proba(texts)

    # Find the highest confidence score for each sample
    max_probas = np.max(proba, axis=1)

    # Identify indices of incorrect predictions
    incorrect_predictions_idx = np.where(y_pred != answer)[0]
    # Extract confidence scores for incorrect predictions
    incorrect_confidences = max_probas[incorrect_predictions_idx]

    # Combine incorrect prediction indices with their confidence scores and sort by confidence
    sorted_incorrect_by_confidence = sorted(zip(incorrect_predictions_idx, incorrect_confidences), key=lambda x: x[1])

    # Get top indices with lowest confidence
    top_incorrect_idx = [idx for idx, _ in sorted_incorrect_by_confidence[:args.sample_size]]

    # Extract complete records for these indices from the original dataset
    file_path = args.test_data_path
    ds = load_dataset('json', data_files=file_path)
    low_confidence_full_data = [ds['train'][int(i)] for i in top_incorrect_idx]

    # Save to JSON file
    filename = f'multinomialnb_low_confidence.json'
    with open(filename, 'w') as f:
        json.dump(low_confidence_full_data, f)


if __name__ == '__main__':
    multinomialnb()

