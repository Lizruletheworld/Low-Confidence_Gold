# model.py

import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import torch
class BertClassify(nn.Module):
    def __init__(self, n_class):
        super(BertClassify, self).__init__()
        self.n_class = n_class
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True, return_dict=True)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = output.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.linear(cls_output)
        return logits

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
