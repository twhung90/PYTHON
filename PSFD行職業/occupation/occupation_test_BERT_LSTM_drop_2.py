# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:36:36 2025

@author: Tsung-wei
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, PreTrainedModel, BertConfig
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
#from occupatiob_train_BERT_LSTM_drop_2 import BertLSTMClassifier  # 從訓練檔載入模型定義

# 定義 BertLSTMClassifier (或從訓練檔載入)
class BertLSTMClassifier(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, bert_model_name, num_labels, hidden_size1=1600, hidden_size2=800, dropout=0.3):
        config = BertConfig.from_pretrained(bert_model_name)
        super().__init__(config)

        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_model_name)
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
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        lstm_out1, _ = self.lstm1(bert_output.last_hidden_state)
        lstm_out2, _ = self.lstm2(lstm_out1)
        logits = self.fc(self.dropout(lstm_out2[:, -1, :]))
        return logits

    def save_pretrained(self, save_directory):
        import os
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.bert.save_pretrained(save_directory)
        torch.save({
            'lstm1': self.lstm1.state_dict(),
            'lstm2': self.lstm2.state_dict(),
            'fc': self.fc.state_dict(),
            'dropout': self.dropout.state_dict()
        }, f"{save_directory}/occupations_BERT_LSTM_model.pth")
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_custom_pretrained(cls, save_directory, num_labels, *args, **kwargs):
        config = BertConfig.from_pretrained(save_directory, **kwargs)
        model = cls(bert_model_name=save_directory, num_labels=num_labels, *args, **kwargs)
        weights = torch.load(f"{save_directory}/occupations_BERT_LSTM_model.pth")
        model.lstm1.load_state_dict(weights['lstm1'])
        model.lstm2.load_state_dict(weights['lstm2'])
        model.fc.load_state_dict(weights['fc'])
        model.dropout.load_state_dict(weights['dropout'])
        return model


# 加载模型
model_path = "occupations_BERT_LSTM_model"
tokenizer_path = "occupations_BERT_LSTM_tokens_saved"
label_classes_path = "label_classes_BERT_LSTM.npy"

# 加载標籤
label_classes = np.load(label_classes_path, allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes
num_labels = len(label_classes)

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertLSTMClassifier.from_custom_pretrained(model_path, num_labels=num_labels)
model.to(device)
model.eval()

# 定义预测函数
def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    predictions = torch.argmax(outputs, dim=-1).item()
    return predictions

# 5. 預測新文本
new_texts = [
    "建碩企業 混泥土載運 運輸 司機",
    "移民署 證照查驗 國境事務大隊 科員 證照查驗",
    "中央研究院 研究員 社會學",
    "中央研究院 統計科學研究所 博士後研究員",
    "司法院 大法官 憲法法庭解釋憲法",
    "玫瑰天主堂 神父 為上帝服務 造福群眾",
    "海軍教準部 教育訓練計畫 訓練 中校 計畫",
    "陸軍 砲兵連 下士",
    "士官長 督導修理飛機人員",
    "台南市政府社會局 老人保護 推廣老年福利政策",
    "HowFun工作室 Youtuber 拍攝業配行銷廣告",
    "中和國小 數學老師",
    "麗山高中 老師 教英文",
    "省鳳中 歷史老師 教學",
    "冠群文教 老師 教英文",
    "冠群文教 老師 物理",
    "勝利補習班 老師 教物理化學",
    "隆龍生命禮儀社 大體修復師 遺體淨身及縫補化妝",
    "台灣鐵路管理局 便當文創商品 交通運輸 站務佐理 剪收票 售票 旅客諮詢",
    "八大行業 陪客人喝酒",
    "路邊停車開單員 路邊停車開單",
    "停車場 老闆 管理",
    "玄元道場 道士 法事",
    "國防部 上將",
    "指紋辨識 IC 經理",
    "酒促 賣酒",
    "高雄地檢署 檢察長 辦理案件",
    "農 開拖引機",
    "桃園航勤 艙服組 作業員",
    "全世好汽車修理廠，技工，修車",
    "飛宏駕訓班 教練 教開車",
    "無 做紙紮屋 紙紮",
    "無印良品 櫃台 收銀，提供客人諮詢服務",
    "六合彩組頭 無 做莊",
    "行政院 院長 統整行政院業務",
    "法國翻譯學會 秘書長 執行與統籌學會相關業務",
    "法國翻譯學會 秘書 幫助執行長處理其業務",
    "南港區農會 辦事員 總務",
    "南港區農會 股員 儲匯業務",
    "理髮師 剪頭髮",
    "XX花園 園丁保安 維護花園安全",
    "杜雷斯 技術員 製作保險套"
]

for text in new_texts:
    predicted_index = predict(text)  # 獲取預測的類別索引
    decoded_labels = label_encoder.inverse_transform([predicted_index])  # 將索引轉換為原始標籤
    print(f"職業: {text} -> 預測類別: {decoded_labels[0]}")  # 輸出預測結果
