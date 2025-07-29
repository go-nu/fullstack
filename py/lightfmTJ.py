import pandas as pd
import numpy as np
from datetime import datetime
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

# ✅ 데이터 로딩 함수 (SQLAlchemy 사용)
def load_data():
    print("\U0001F4E6 데이터 로딩 중...")
    engine = create_engine("mysql+mysqlconnector://root:1234@localhost:3306/woorizip")

    log_df = pd.read_sql("""
        SELECT user_id, model_id, product_id, weight, timestamp
        FROM recommend_log
    """, engine)
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp']).apply(lambda x: int(x.timestamp()))

    user_df = pd.read_sql("""
        SELECT id AS user_id, gender, birth, residence_type
        FROM users
    """, engine)
    user_df['age_group'] = 1  # 하나의 나이대로 고정

    model_attr_df = pd.read_sql("""
        SELECT pma.product_model_id AS model_id, a.name AS attribute_type, av.value AS attribute_value
        FROM product_model_attribute pma
        JOIN attribute_value av ON pma.attribute_value_id = av.id
        JOIN attribute a ON av.attribute_id = a.id
    """, engine)
    model_pivot = model_attr_df.pivot_table(index='model_id', columns='attribute_type', values='attribute_value', aggfunc='first').reset_index()

    df = log_df.merge(user_df, on='user_id', how='left') \
               .merge(model_pivot, on='model_id', how='left')

    df.dropna(subset=['user_id', 'model_id', 'weight', 'gender', 'residence_type', 'age_group', '색상', '사이즈', '소재'], inplace=True)

    # 문자열로 인코딩 (LightFM용)
    for col in ['user_id', 'model_id', 'gender', 'residence_type', '색상', '사이즈', '소재']:
        df[col] = df[col].astype(str)

    return df

# ✅ main
if __name__ == '__main__':
    df = load_data()

    # ✅ LightFM Dataset 생성
    dataset = Dataset()
    dataset.fit(users=df['user_id'], items=df['model_id'])

    (interactions, weights) = dataset.build_interactions(((row['user_id'], row['model_id'], row['weight']) for _, row in df.iterrows()))

    # ✅ 모델 학습
    model = LightFM(loss='warp')
    model.fit(interactions, sample_weight=weights, epochs=10, num_threads=1)

    # ✅ 평가
    precision = precision_at_k(model, interactions, k=5).mean()
    auc = auc_score(model, interactions).mean()

    print(f"\n📊 Precision@5: {precision:.4f}")
    print(f"📊 AUC Score: {auc:.4f}")
