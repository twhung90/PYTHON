# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:36:36 2025

@author: Tsung-wei
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. 載入訓練好的模型和分詞器
model_path = "industry_BERT_model"  # 訓練好的模型路徑
tokenizer_path = "industry_BERT_tokens_saved"  # 分詞器路徑

# 加載分詞器
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# 加載訓練好的BERT模型
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # 設置模型為評估模式

# 2. 定義預測函數
def predict(text):
    # 對輸入文本進行分詞處理
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # 將數據移動到GPU（如果可用）
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():  # 禁用梯度計算
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits  # 獲取logits
    predictions = torch.argmax(logits, dim=-1)  # 獲取預測結果
    
    return predictions.item()  # 返回預測的標籤

# 加載訓練時保存的標籤類別
label_classes = np.load('industry_label_classes_BERT.npy', allow_pickle=True)  # 加載標籤類別
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# 3. 測試新文本的預測
new_texts = [
    "茶的魔手茶飲 茶飲販售",
    "陸軍 國防服務",
    "沙龍 美髮",
    "火鍋店 火鍋",
    "第一保全 守衛",
    "無	個人家教",
    "宸全安全工程有限公司	生產製造營造工廠",
    "鼎業旅行社	訂房網站",
    "世鑫食品股份有限公司	生產製造罐頭",
    "無	銷售炸物類 雞排等等",
    "7-11統一超商 銷售食物，零售",
    "Foodpanda 食品外送",
    "小籠包 賣小籠包",
    "IT軟體設計公司",
    "神岡國中 教育學生",
    "新建興五金行 銷售五金用品",
    "路益交通公司 工地建材品項載運",
    "環協工程公司 噪音污染防治",
    "防噪音施工服務",
    "綠地農場 種鳳梨",
    "保母",
    "褓姆",
    "酒吧",
    "室內設計",
    "公益彩券投注站",
    "農田水利會",
    "水利會",
    "家管",
    "家庭主夫",
    "不知道"
]

for text in new_texts:
    predicted_index = predict(text)  # 獲取預測的類別索引
    decoded_labels = label_encoder.inverse_transform([predicted_index])  # 將索引轉換為原始標籤
    print(f"行業: {text} -> 預測類別: {decoded_labels[0]}")  # 輸出預測結果
