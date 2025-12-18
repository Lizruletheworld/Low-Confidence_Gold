from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from datasets import load_dataset
import json
import os

# 建议保留此函数用于其他用途，但在 clustering 流程中不需要它读取主文件
def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clustering(file_path):
    print(f"正在加载模型并处理文件: {file_path}")
    
    # 1. 加载模型（之前已下载完成，这次会很快）
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # 2. 正确加载数据集：直接传入路径字符串
    ds = load_dataset('json', data_files=file_path)
    data = pd.DataFrame(ds['train'])

    # 3. 文本编码
    print("正在进行文本向量化 (Encoding)... 这可能需要一点时间")
    texts = [str(inst) + ' ' + str(inp) for inst, inp in zip(ds['train']['instruction'], ds['train']['input'])]
    embeddings = model.encode(texts, show_progress_bar=True)

    # 4. PCA 降维
    pca = PCA(n_components=0.95)
    reduced_embeddings = pca.fit_transform(embeddings)

    # 5. 聚类
    n = len(reduced_embeddings)
    k = max(1, int(np.sqrt(n / 8))) # 确保 k 至少为 1
    print(f"检测到样本量: {n}, 自动设定类簇数 k: {k}")

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(reduced_embeddings)

    data['cluster'] = clusters
    
    # 保存带标签的完整数据
    data.to_json('alpaca_labels.json', orient='records', lines=True, indent=4)

    # 6. 筛选每个类簇中靠近中心的代表性样本
    distances = kmeans.transform(reduced_embeddings)
    closest_texts_data_list = []
    closest_dataset_list = []

    for i in range(k):
        # 获取属于当前类簇的索引
        cluster_mask = (clusters == i)
        cluster_distances = distances[cluster_mask, i]
        
        # 排序并取前 40 个
        sorted_indices = cluster_distances.argsort()
        closest_indices = sorted_indices[:min(40, len(sorted_indices))]
        
        # 提取数据
        cluster_data = data[cluster_mask].iloc[closest_indices].copy()
        cluster_data['cluster'] = i
        
        closest_texts_data_list.append(cluster_data[['instruction', 'input', 'cluster']])
        closest_dataset_list.append(cluster_data[['instruction', 'input', 'output']])

    # 合并结果
    train_dataset = pd.concat(closest_texts_data_list, ignore_index=True)
    closest_dataset = pd.concat(closest_dataset_list, ignore_index=True)

    # 导出
    train_dataset.to_json('train_dataset.json', orient='records', force_ascii=False)
    closest_dataset.to_json('closest_dataset.json', orient='records', force_ascii=False)
    
    print("✅ 处理完成！已生成 train_dataset.json 和 closest_dataset.json")
    return train_dataset, closest_dataset

if __name__ == '__main__':
    # 修正路径逻辑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设你的数据在项目根目录的 data 文件夹下
    json_path = os.path.join(current_dir, '..', 'data', 'alpaca_data_cleaned.json')

    if os.path.exists(json_path):
        # 🌟 关键：直接传入路径字符串，不要调用 load_json_file
        clustering(json_path)
    else:
        print(f"❌ 错误：在以下路径找不到数据文件: {json_path}")

