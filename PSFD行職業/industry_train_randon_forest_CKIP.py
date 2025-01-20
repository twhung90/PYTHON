# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:41:42 2024

@author: Tsung-wei
"""

import random
from ckiptagger import WS, POS, NER
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib

# 1. 準備數據集
dt = pd.read_csv("D:\\Users\\survey\\Desktop\\PSFD行職業\\data\\PSFD_2018_industry_U2.csv")
texts = dt['text'].tolist() 

labels = dt['indust_n'].tolist()  # 對應的標籤

# 初始化 CKIP Tagger，指定模型路徑
ws = WS(".\\CKIP\\data")  # 替換為正確的 CKIP 模型路徑
pos = POS(".\\CKIP\\data")  # 如果需要詞性標註
ner = NER(".\\CKIP\\data")  # 如果需要命名實體識別

word_slices = ws(texts)

# 2. 將文本轉換為嵌入
sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = []
for vectors in word_slices:
    # 將斷詞結果轉換為句子格式
    sentence = " ".join(vectors)
    # 獲取句子嵌入
    embedding = sbert_model.encode(sentence)
    embeddings.append(embedding)

# 3. 將標籤編碼為數字
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)    #將labels轉換為數值

seed_value = 7575
np.random.seed(seed_value)      # NumPy 隨機種子
random.seed(seed_value)         # Python 隨機種子

# 4. 拆分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.01, random_state=seed_value)


# 5. 訓練隨機森林分類器
random_forest_model = RandomForestClassifier(
    n_estimators=2500,       # 決策樹的數量，提高穩定性
    max_depth=45,           # 設定合理深度，防止過擬合
    min_samples_split=5,   # 增加分裂所需最小樣本數，控制模型複雜度
    min_samples_leaf=5,     # 限制葉節點最小樣本數
    random_state=seed_value # 隨機種子，使隨機性一致
)

# 6.訓練資料
random_forest_model.fit(X_train, y_train)  # 用訓練集訓練模型

# 6. 測試模型並評估準確率
y_pred = random_forest_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy:.4f}')

'''
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")
'''

# 7. 保存模型
joblib.dump(random_forest_model, 'industry_random_forest_model.joblib')  # 保存joblib模型
joblib.dump(random_forest_model, 'industry_random_forest_model.pkl')  # 保存pickle模型
