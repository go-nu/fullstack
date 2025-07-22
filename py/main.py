import mysql.connector
import pandas as pd
from lightfm.data import Dataset

# DB 연결 및 데이터 조회
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="1234",
    database="woorizip"
)
cursor = conn.cursor()
cursor.execute("SELECT user_id, product_id, model_id, weight, timestamp FROM recommend_log")
logs = cursor.fetchall()
cursor.close()
conn.close()

# DataFrame 생성
df = pd.DataFrame(logs, columns=['user_id', 'product_id', 'model_id', 'weight', 'timestamp'])

# 전처리
df['timestamp'] = df['timestamp'].apply(lambda x: int(x.timestamp()))
df['model_id'] = df['model_id'].fillna(0).astype(int)

# ✅ Interaction Matrix 생성
dataset = Dataset()
dataset.fit(
    users=df['user_id'].unique(),
    items=df['model_id'].unique()
)

interactions_data = [
    (row['user_id'], row['model_id'], row['weight']) for _, row in df.iterrows()
]

(interactions, weights) = dataset.build_interactions(interactions_data)

# 결과 출력
print("Interaction matrix shape:", interactions.shape)
print("Sample interactions (non-zero entries):")
print(interactions)
