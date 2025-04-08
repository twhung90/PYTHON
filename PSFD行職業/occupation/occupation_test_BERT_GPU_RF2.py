# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 18:42:17 2025

@author: Tsung-wei
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# 1. 載入模型與 tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("occupations_BERT_RF2_model").to(device)
tokenizer = BertTokenizer.from_pretrained("occupations_BERT_RF2_tokens_saved")
rf = joblib.load("occupations_BERT_RF2_model.joblib")
label_classes = np.load("label_classes_BERT_RF2.npy", allow_pickle=True)


# 2. 測試新文本的預測
texts = [
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
    "XX生命 帶領家屬拜拜",
    "台灣鐵路管理局 便當文創商品 交通運輸 站務佐理 剪收票 售票 旅客諮詢",
    "八大行業 陪客人喝酒",
    "台灣中油 工讀生 加油",
    "路邊停車開單員 路邊停車開單",
    "玄元道場 道士 法事",
    "國防部 部長 一級上將",
    "夜店 播放唱片",
    "XX馬戲團 腹語師 表演腹語",
    "農 農夫 種蓮藕",
    "淋巴排毒 服務員 幫助客人放鬆 排毒",
    "XX酒店 服務部 代客泊車",
    "年代 新聞 記者 採訪",
    "檸檬家事服務 家庭清潔",
    "OKAMOTO 作業員 生產保險套"
    ]

# 3. 預處理並轉換為張量
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)


# 4. 模型推論 & 建構 RF 特徵
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    top3 = torch.topk(outputs.logits, k=3, dim=-1)
    indices = top3.indices.float()
    scores = top3.values
    features = torch.cat([indices, scores], dim=-1).cpu().numpy()  # shape = [batch_size, 6]


# 5. 使用 RF 預測職業編碼，轉回職業名稱
preds = rf.predict(features)
predicted_classes = label_classes[preds]


# 6. 顯示結果
for i, text in enumerate(texts):
    print(f"\n職業：{text}\n→ 預測類別：{predicted_classes[i]}")