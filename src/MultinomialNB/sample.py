from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score
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

if __name__ == '__main__':
    multinomialnb()
