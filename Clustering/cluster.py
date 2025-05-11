
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from datasets import load_dataset
import json
import os

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clustering(file_path):

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    ds = load_dataset('json', data_files=file_path)
    data = pd.DataFrame(ds['train'])

    texts = [input_text + ' ' + instruction_text for input_text, instruction_text in zip(ds['train']['input'], ds['train']['instruction'])]
    embeddings = model.encode(texts)

    pca = PCA(n_components=0.95)
    reduced_embeddings = pca.fit_transform(embeddings)

    n = len(reduced_embeddings)
    k = int(np.sqrt(n / 8))

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reduced_embeddings)

    clusters = kmeans.labels_

    data['cluster'] = clusters

    new_file_path ='alpaca_labels.json'

    data.to_json(new_file_path, orient='records', lines=True, indent=4)

    distances = kmeans.transform(reduced_embeddings)


    closest_dataset= pd.DataFrame()
    closest_texts_data_list = []

    for i in range(k):

        cluster_distances = distances[clusters == i, i]

        sorted_indices = cluster_distances.argsort()

        closest_indices = sorted_indices[:min(40, len(sorted_indices))]
        closest_texts = data.iloc[clusters == i].iloc[closest_indices]

        closest_texts['cluster'] = i  # 为当前簇设置正确的簇编号
        closest_texts_data_list.append(closest_texts[[ 'instruction','input', 'cluster']])  # 保存当前簇的数据

        closest_dataset = pd.concat([closest_dataset, closest_texts[[ 'instruction', 'input','output']]], ignore_index=True)
    # 重置索引
    train_dataset = pd.concat(closest_texts_data_list, ignore_index=True)
    train_dataset.reset_index(drop=True, inplace=True)
    closest_dataset.reset_index(drop=True, inplace=True)

    train_dataset.to_json('train_dataset.json', orient='records')
    closest_dataset.to_json('closest_dataset.json', orient='records')


    return train_dataset, closest_dataset


if __name__ == '__main__':

    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, '..', 'data', 'alpaca_data_cleaned.json')
    file_path = load_json_file(json_path)

    clustering(file_path)

