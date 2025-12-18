import torch.nn as nn
from transformers import DistilBertModel
import torch

class BertClassify(nn.Module):
    def __init__(self, n_class):
        super(BertClassify, self).__init__()
        self.n_class = n_class
        # DistilBERT 不需要 token_type_ids，设置保持默认即可
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True, return_dict=True)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        # --- 修改点：移除 token_type_ids 传参 ---
        # DistilBertModel 的 forward 不接受 token_type_ids
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # DistilBERT 的 output 是一个 BaseModelOutput 对象
        # last_hidden_state 形状: (batch_size, sequence_length, 768)
        last_hidden_state = output.last_hidden_state
        
        # 提取 [CLS] token 的向量 (索引为 0)
        cls_output = last_hidden_state[:, 0, :]
        
        cls_output = self.dropout(cls_output)
        logits = self.linear(cls_output)
        return logits

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        # 添加 map_location 以确保在 CPU/GPU 切换时不会报错
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=device))
