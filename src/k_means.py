
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

    # 加载模型
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    ds = load_dataset('json', data_files=file_path)
    data = pd.DataFrame(ds['train'])

    # 使用列表推导式和zip函数来连接input和instruction字段
    texts = [input_text + ' ' + instruction_text for input_text, instruction_text in zip(ds['train']['input'], ds['train']['instruction'])]
    embeddings = model.encode(texts)


    # 使用PCA进行降维
    pca = PCA(n_components=0.95)
    reduced_embeddings = pca.fit_transform(embeddings)

    n = len(reduced_embeddings)
    k = int(np.sqrt(n / 8))

    # 初始化KMeans聚类器
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reduced_embeddings)

    # 获取聚类结果
    clusters = kmeans.labels_

    # 将聚类结果添加到原始数据框中
    data['cluster'] = clusters
    # 现在data包含了原始数据以及聚类结果
    new_file_path ='E:/alpaca/new_data_k_40_40.json'

    data.to_json(new_file_path, orient='records', lines=True, indent=4)
    # 计算每个点到簇中心的距离
    distances = kmeans.transform(reduced_embeddings)

    # 初始化数据集
    closest_texts_input_output = pd.DataFrame()
    closest_texts_data_list = []  # 创建一个列表来保存每个簇的数据
    # 提取每个簇最近的文本
    for i in range(k):
        # 获取当前簇到中心点的距离
        cluster_distances = distances[clusters == i, i]

        # 对距离进行排序并获取索引
        sorted_indices = cluster_distances.argsort()

        closest_indices = sorted_indices[:min(40, len(sorted_indices))]
        closest_texts = data.iloc[clusters == i].iloc[closest_indices]

        # 创建一个新的DataFrame来保存当前簇的数据
        closest_texts['cluster'] = i  # 为当前簇设置正确的簇编号
        closest_texts_data_list.append(closest_texts[[ 'instruction','input', 'cluster']])  # 保存当前簇的数据

        closest_texts_input_output = pd.concat([closest_texts_input_output, closest_texts[[ 'instruction', 'input','output']]], ignore_index=True)
    # 重置索引
    closest_texts_data = pd.concat(closest_texts_data_list, ignore_index=True)
    closest_texts_data.reset_index(drop=True, inplace=True)
    closest_texts_input_output.reset_index(drop=True, inplace=True)

    closest_texts_data.to_json('E:/alpaca/closest_texts_data_k_40_40.json', orient='records')
    closest_texts_input_output.to_json('E:/alpaca/closest_texts_input_output_k_40_40.json', orient='records')


    return closest_texts_data, closest_texts_input_output


if __name__ == '__main__':

    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, '..', 'data', 'alpaca_data_cleaned.json')
    file_path = load_json_file(json_path)

    clustering(file_path)

