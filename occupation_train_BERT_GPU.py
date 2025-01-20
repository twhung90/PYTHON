# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:53:35 2024

@author: Tsung-wei
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


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

# 3. 加載BERT模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 檢查是否有可用的GPU
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels)  # 根據類別數調整
model.to(device)  # 將模型移動到GPU

# 4. 數據處理函數
def encode_data(texts, encoded_labels):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs['labels'] = torch.tensor(encoded_labels)  # 使用 'labels' 作為鍵名以匹配模型輸入
    return inputs

class TextDataset(Dataset):
    def __init__(self, texts, encoded_labels):
        self.inputs = encode_data(texts, encoded_labels)

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx].to(device) for key, val in self.inputs.items()}  # 將數據移動到GPU

# 創建數據集並劃分訓練集和測試集（80%訓練，20%測試）
full_dataset = TextDataset(texts, encoded_labels)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 創建dataloader
train_loader = DataLoader(train_dataset, batch_size=32)  # 根據需要調整批量大小
test_loader = DataLoader(test_dataset, batch_size=32)

# 5. 模型訓練
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 使用較小的學習率

model.train()
num_epochs = 25  # 可以根據需要調整訓練輪數
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in train_loader:
        optimizer.zero_grad()  # 清除梯度
        outputs = model(**batch)  # 前向傳播
        loss = outputs.loss  # 獲取損失值
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新參數
        
        total_loss += loss.item()  # 累加損失值
        
        # 計算正確預測的數量
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct = (predictions == batch['labels']).sum().item()  # 計算正確預測的數量
        total_correct += correct
        total_samples += batch['labels'].size(0)  # 總樣本數
        
        accuracy = total_correct / total_samples  # 計算準確率
        avg_loss = total_loss / len(train_loader)  # 計算平均損失
    
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.8f}, Accuracy: {accuracy:.4f}")  # 輸出損失值和準確率

# 6. 模型測試
model.eval()
total_correct = 0
total_samples = 0

for batch in test_loader:
    with torch.no_grad():
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # 計算正確預測的數量
        correct = (predictions == batch['labels']).sum().item()  # 計算正確預測的數量
        total_correct += correct
        total_samples += batch['labels'].size(0)  # 總樣本數

# 7. 計算準確率
accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.4f}")  # 輸出測試準確率


# 8. 保存模型
model.save_pretrained("occupations_BERT_model")  # 保存模型
tokenizer.save_pretrained("occupations_BERT_tokens_saved")  # 保存分詞器
np.save('label_classes_BERT.npy', label_encoder.classes_)  # 保存labels類別