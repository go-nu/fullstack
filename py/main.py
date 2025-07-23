import pandas as pd
import mysql.connector
from datetime import datetime

# ✅ 1. MySQL 연결
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="1234",
    database="woorizip"
)

# ✅ 2. 추천 로그 로딩
log_query = """
SELECT user_id, model_id, product_id, weight, timestamp
FROM recommend_log
"""
log_df = pd.read_sql(log_query, conn)

# ✅ 2-1. timestamp → Unix timestamp로 변환
log_df['timestamp'] = pd.to_datetime(log_df['timestamp']).apply(lambda x: int(x.timestamp()))

# ✅ 3. 사용자 정보 로딩
user_query = """
SELECT id AS user_id, gender, birth, residence_type
FROM users
"""
user_df = pd.read_sql(user_query, conn)

# ✅ 3-1. 사용자 나이 계산 후 그룹화
current_year = datetime.now().year
user_df['age'] = current_year - pd.to_datetime(user_df['birth']).dt.year
user_df['age_group'] = pd.cut(
    user_df['age'],
    bins=[-1, 9, 19, 29, 39, 49, 150],
    labels=[0, 1, 2, 3, 4, 5],
    right=True
).astype(int)
user_df.drop(['birth', 'age'], axis=1, inplace=True)

# ✅ 4. 상품 정보 로딩
product_query = """
SELECT id AS product_id, category_id
FROM product
"""
product_df = pd.read_sql(product_query, conn)

# ✅ 5. 모델 속성 로딩
model_query = """
SELECT 
    pma.product_model_id AS model_id,
    a.name AS attribute_type,
    av.value AS attribute_value
FROM product_model_attribute pma
JOIN attribute_value av ON pma.attribute_value_id = av.id
JOIN attribute a ON av.attribute_id = a.id
"""
model_df = pd.read_sql(model_query, conn)

# ✅ 6. 피벗 처리 (속성 컬럼: 색상, 사이즈, 소재 등)
model_pivot = model_df.pivot_table(
    index='model_id',
    columns='attribute_type',
    values='attribute_value',
    aggfunc='first'
).reset_index()

# ✅ 7. 병합
merged = log_df \
    .merge(user_df, on='user_id', how='left') \
    .merge(product_df, on='product_id', how='left') \
    .merge(model_pivot, on='model_id', how='left')

# ✅ 8. 저장
merged.to_csv("recommend_data_full.csv", index=False, encoding="utf-8-sig")
print("✅ recommend_data_full.csv 저장 완료 (Unix timestamp 적용)")
