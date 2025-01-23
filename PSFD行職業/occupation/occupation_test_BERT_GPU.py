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
model_path = "occupations_BERT_model"  # 訓練好的模型路徑
tokenizer_path = "occupations_BERT_tokens_saved"  # 分詞器路徑

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
    
    with torch.no_grad():  # 禁用梯度計算以節省內存和加速計算
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits  # 獲取logits
    predictions = torch.argmax(logits, dim=-1)  # 獲取預測結果
    
    return predictions.item()  # 返回預測的標籤

# 加載訓練時保存的標籤類別
label_classes = np.load('label_classes_BERT.npy', allow_pickle=True)  # 加載標籤類別
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# 3. 測試新文本的預測
new_texts = [
    "建碩企業 混泥土載運 運輸 司機",
    "移民署 證照查驗 國境事務大隊 科員 證照查驗",
    "中央研究院 研究員 社會學",
    "中央研究院 統計科學研究所 博士後研究員",
    "司法院 大法官 憲法法庭解釋憲法",
    "玫瑰天主堂 神父 為上帝服務 造福群眾",
    "海軍教準部 教育訓練計畫 訓練 中校 計畫",
    "士官長 督導修理飛機人員",
    "台南市政府社會局 老人保護 推廣老年福利政策",
    "HowFun工作室 Youtuber 拍攝網路業配影片",
    "油土伯 拍搞笑影片",
    "中和國小 數學老師",
    "省鳳中 歷史老師 教學",
    "師大附中 老師 教國文",
    "冠群文教 老師 教英文",
    "勝利補習班 老師 教物理化學",
    "隆龍生命禮儀社 大體修復師 遺體淨身及縫補化妝",
    "台灣鐵路管理局 便當文創商品 交通運輸 站務佐理 剪收票 售票 旅客諮詢",
    "八大行業 陪客人喝酒",
    "台灣中油 工讀生 加油",
    "路邊停車開單員 路邊停車開單",
    "玄元道場 道士 法事",
    "國防部 部長 一級上將",
    "夜店 播放唱片",
    "XX馬戲團 腹語師 表演腹語",
    "農 農夫 種蓮藕"
]

for text in new_texts:
    predicted_index = predict(text)  # 獲取預測的類別索引
    decoded_labels = label_encoder.inverse_transform([predicted_index])  # 將索引轉換為原始標籤
    print(f"職業: {text} -> 預測類別: {decoded_labels[0]}")  # 輸出預測結果
