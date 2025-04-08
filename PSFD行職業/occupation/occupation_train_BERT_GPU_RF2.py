# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 18:42:17 2025

@author: Tsung-wei
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import joblib

# 1. 讀取數據集
data_path = "D:\\Users\\survey\\Desktop\\PSFD行職業\\data\\PSFD_2018-20_occupations_全UN.csv"
dt = pd.read_csv(data_path)
texts = dt['text'].tolist()
labels = dt['occu_n'].tolist()

# 2. 標籤編碼
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(set(encoded_labels))

# 設定隨機種子
seed_value = 7575
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# 3. 加載BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels).to(device)

# 4. 自定義 Dataset
class TextDataset(Dataset):
    def __init__(self, texts, encoded_labels):
        self.inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.inputs['labels'] = torch.tensor(encoded_labels)
    
    def __len__(self):
        return len(self.inputs['input_ids'])
    
    def __getitem__(self, idx):
        return {key: val[idx].to(device) for key, val in self.inputs.items()}

# 數據集與劃分
full_dataset = TextDataset(texts, encoded_labels)
train_size = int(0.95 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 5. 模型訓練
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()
num_epochs = 25
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        total_correct += (preds == batch['labels']).sum().item()
        total_samples += batch['labels'].size(0)

    print(f"[Epoch {epoch+1}]  Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_correct/total_samples:.4f}")
    

# 進行繼續訓練 (從 epoch=25 訓練到 epoch=30)
model.train()
num_epochs = 3  # 額外訓練 5 個 epoch
for epoch in range(26, 29):
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(**batch)  # 前向傳播
        #outputs = model(inputs['input_ids'], inputs['attention_mask'])  # 前向傳播
        loss = outputs.loss  # 獲取損失值
        #loss = nn.CrossEntropyLoss()(outputs, labels)  # 獲取損失值
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        
    accuracy = total_correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch}]  Loss: {avg_loss:.8f}, Accuracy: {accuracy:.4f}")

# 6. 提取 features (Top-3 index + score) 用於 RF 訓練
model.eval()
rf_features = []
rf_labels = []

with torch.no_grad():
    for batch in train_loader:
        outputs = model(**batch)
        top3 = torch.topk(outputs.logits, k=3, dim=-1)
        
        indices = top3.indices.float()  # 獲取各類別的所在 index
        scores = top3.values  # 取出實際的 logits 分數
        
        features = torch.cat([indices, scores], dim=-1)  # shape: [batch_size, 6]
        rf_features.append(features.cpu().numpy())
        rf_labels.append(batch['labels'].cpu().numpy())

X_rf = np.vstack(rf_features)
y_rf = np.concatenate(rf_labels)

rf = RandomForestClassifier(n_estimators=100, random_state=seed_value)
rf.fit(X_rf, y_rf)
joblib.dump(rf, './occupations_BERT_RF2_model.joblib')


# 7. 測試模型
test_features = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        top3 = torch.topk(outputs.logits, k=3, dim=-1)
        indices = top3.indices.float()
        scores = top3.values
        features = torch.cat([indices, scores], dim=-1)
        test_features.append(features.cpu().numpy())
        test_labels.append(batch['labels'].cpu().numpy())

x_test_rf = np.vstack(test_features)
y_test_rf = np.concatenate(test_labels)

# 使用 RF 預測
rf = joblib.load('./occupations_BERT_RF2_model.joblib')
rf_preds = rf.predict(x_test_rf)
accuracy = (rf_preds == y_test_rf).mean()
print(f"\nRandom Forest Final Accuracy: {accuracy:.4f}")


# 8. 額外評估：BERT Top-3 Accuracy
top3_hits = 0
total_samples = 0

with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        top3_preds = torch.topk(outputs.logits, k=3, dim=-1).indices
        for i in range(len(batch['labels'])):
            if batch['labels'][i] in top3_preds[i]:
                top3_hits += 1
        total_samples += len(batch['labels'])

print(f"BERT Top-3 Accuracy: {top3_hits / total_samples:.4f}")

# 9. 儲存模型與相關物件
model.save_pretrained("occupations_BERT_RF2_model")
tokenizer.save_pretrained("occupations_BERT_RF2_tokens_saved")
np.save('label_classes_BERT_RF2.npy', label_encoder.classes_)