# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:53:35 2024

@author: Tsung-wei
"""

import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, PreTrainedModel, BertConfig
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

# 1. 準備數據集
data_path = "D:\\Users\\survey\\Desktop\\PSFD行職業\\data\\PSFD_2018-20_occupations_全U.csv"
dt = pd.read_csv(data_path)
texts = dt['text'].tolist()  # 文本數據
labels = dt['occu_n'].tolist()  # 標籤

# 2. 將標籤編碼為數字
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(set(encoded_labels))  # 計算唯一標籤的數量

# 設置隨機種子
seed_value = 7575
torch.manual_seed(seed_value)   # PyTorch 隨機種子
np.random.seed(seed_value)      # NumPy 隨機種子
random.seed(seed_value)         # Python 隨機種子

# 3. 自定義BERT + LSTM 的分類器
class BertLSTMClassifier(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, bert_model_name, num_labels, hidden_size1=1600, hidden_size2=800, dropout=0.3):
        # 从预训练模型名加载配置
        config = BertConfig.from_pretrained(bert_model_name)
        super().__init__(config)

        self.num_labels = num_labels
        
        # 加载 BERT 模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 自定义 LSTM 和分类层
        self.lstm1 = nn.LSTM(input_size=config.hidden_size, 
                             hidden_size=hidden_size1, 
                             batch_first=True,
                             bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1 * 2, 
                             hidden_size=hidden_size2, 
                             batch_first=True,
                             bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2 * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 透過 BERT 獲取隱藏曾狀態
        bert_output = self.bert(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids)
        
        lstm_out1, _ = self.lstm1(bert_output.last_hidden_state)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # 分類層输出
        out = self.dropout(lstm_out2[:, -1, :])
        logits = self.fc(out)
        return logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 從預先訓練模型加载 BERT 配置
        config = BertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)

    def save_pretrained(self, save_directory):
        # 保存 BERT 配置和LSTM模型权重
        self.bert.save_pretrained(save_directory)
        
        # 保存 LSTM 模型 + 分類器權重
        torch.save({
            'lstm1': self.lstm1.state_dict(),
            'lstm2': self.lstm2.state_dict(),
            'fc': self.fc.state_dict(),
            'dropout': self.dropout.state_dict()
        }, f"{save_directory}/occupations_BERT_LSTM_model.pth")   #這裡會儲存LSTM的權重的檔名
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_custom_pretrained(cls, save_directory, *model_args, **kwargs):
        # 加载配置
        config = BertConfig.from_pretrained(save_directory, **kwargs)
        
        # 初始化模型
        model = cls(config=config, *model_args, **kwargs)
        
        # 加载權重
        weights = torch.load(f"{save_directory}/occupations_BERT_LSTM_model.pth")
        model.lstm1.load_state_dict(weights['lstm1'])
        model.lstm2.load_state_dict(weights['lstm2'])
        model.fc.load_state_dict(weights['fc'])
        model.dropout.load_state_dict(weights['dropout'])
        
        return model

# 4. 加載BERT模型到自定義分類器中
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 檢查是否有可用的GPU
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertLSTMClassifier(bert_model_name="bert-base-chinese", num_labels=num_labels)  # 根據類別數調整
model.to(device)  # 將模型移動到GPU

# 5. 數據處理函數
def encode_data(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return inputs

class TextDataset(Dataset):
    def __init__(self, texts, encoded_labels):
        self.inputs = encode_data(texts)
        self.labels = torch.tensor(encoded_labels)

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx].to(device) for key, val in self.inputs.items()}, self.labels[idx].to(device)

# 創建數據集並劃分訓練集和測試集（80%訓練，20%測試）
full_dataset = TextDataset(texts, encoded_labels)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 創建dataloader
train_loader = DataLoader(train_dataset, batch_size=32)  # 根據需要調整批量大小
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. 模型訓練
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 使用較小的學習率

model.train()
num_epochs = 60  # 可以根據需要調整訓練輪數
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader:
        optimizer.zero_grad()  # 清除梯度
        
        inputs, labels = batch
        
        outputs = model(inputs['input_ids'], inputs['attention_mask'])  # 前向傳播
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, labels)  # 獲取損失值
        
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新參數
        
        total_loss += loss.item()  # 累加損失值
        
        # 計算正確預測的數量
        predictions = torch.argmax(outputs, dim=-1)
        correct = (predictions == labels).sum().item()  # 計算正確預測的數量
        total_correct += correct
    
        accuracy = total_correct / len(train_loader.dataset)  # 計算準確率
        avg_loss = total_loss / len(train_loader)  # 計算平均損失
    
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.8f}, Accuracy: {accuracy:.4f}")  # 輸出損失值和準確率
        
# 7. 模型測試
model.eval()
total_correct_test = 0

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        
        predictions = torch.argmax(outputs, dim=-1)
        
        correct = (predictions == labels).sum().item()  # 計算正確預測的數量
        total_correct_test += correct

# 計算準確率
test_accuracy = total_correct_test / len(test_loader.dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")  # 輸出測試準確率


# 模型訓練完成後，您可以保存模型或進行評估
model.save_pretrained("occupations_BERT_LSTM_model")  # 保存模型
tokenizer.save_pretrained("occupations_BERT_LSTM_tokens_saved")  # 保存分詞器
np.save('label_classes_BERT_LSTM.npy', label_encoder.classes_)  # 保存labels類別
