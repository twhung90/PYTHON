# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:18:33 2024

@author: survey
"""


import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from ckiptagger import WS, POS, NER

# 1. 讀取新的 CSV 文件
input_file = 'industry_test.csv'
data = pd.read_csv(input_file)

# 2. 載入隨機森林模型
model = joblib.load('random_forest_model.joblib')

# 載入標籤編碼器
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes_indust.npy', allow_pickle=True)

# 初始化 SBERT 模型
sbert_model = SentenceTransformer('all-mpnet-base-v2')


# 確保文件中包含 'id' 和 'texts' 欄位
if 'id' not in data.columns or 'texts' not in data.columns:
    raise ValueError("CSV 文件必須包含 'id' 和 'texts' 欄位")

# 3. 將文本數據處理為嵌入向量
texts = data['texts'].tolist()

# 使用 CKIP 斷詞
ws = WS(".\\CKIP\\data")  # 替換為正確的 CKIP 模型路徑
pos = POS(".\\CKIP\\data")  # 如果需要詞性標註
ner = NER(".\\CKIP\\data")  # 如果需要命名實體識別
word_slices = ws(texts)

# 使用 SBERT 將文本轉換為嵌入向量
embeddings = [sbert_model.encode(" ".join(vectors)) for vectors in word_slices]

# 4. 預測行業分類
embeddings = np.array(embeddings)
predicted_labels = model.predict(embeddings)

# 解碼類別標籤
decoded_predictions = label_encoder.inverse_transform(predicted_labels)

# 5. 將預測結果加入原始數據框架
data['predictions'] = decoded_predictions

# 6. 將結果保存到新的 CSV 文件
output_file = 'predictions.csv'  # 輸出文件名
data.to_csv(output_file, index=False, encoding='utf-8-sig')
