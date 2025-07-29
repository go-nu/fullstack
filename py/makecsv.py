import pandas as pd
from sqlalchemy import create_engine

# DB 연결 설정
user = 'root'
password = '1234'
host = 'localhost'
port = 3306
database = 'woorizip'

# SQLAlchemy 엔진 생성
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')

# 로그 데이터 불러오기
query = """
    SELECT rl.user_id, rl.model_id, rl.product_id, rl.weight, rl.timestamp,
           u.gender, u.residence_type, 
           CASE
               WHEN TIMESTAMPDIFF(YEAR, u.birth, CURDATE()) < 10 THEN 0
               WHEN TIMESTAMPDIFF(YEAR, u.birth, CURDATE()) < 20 THEN 1
               WHEN TIMESTAMPDIFF(YEAR, u.birth, CURDATE()) < 30 THEN 2
               WHEN TIMESTAMPDIFF(YEAR, u.birth, CURDATE()) < 40 THEN 3
               WHEN TIMESTAMPDIFF(YEAR, u.birth, CURDATE()) < 50 THEN 4
               ELSE 5
           END AS age_group,
           av1.value AS 색상,
           av2.value AS 사이즈,
           av3.value AS 소재
    FROM recommend_log rl
    JOIN users u ON rl.user_id = u.id
    JOIN product_model_attribute pma1 ON rl.model_id = pma1.product_model_id
    JOIN attribute_value av1 ON pma1.attribute_value_id = av1.id AND av1.attribute_id = (SELECT id FROM attribute WHERE name = '색상')
    JOIN product_model_attribute pma2 ON rl.model_id = pma2.product_model_id
    JOIN attribute_value av2 ON pma2.attribute_value_id = av2.id AND av2.attribute_id = (SELECT id FROM attribute WHERE name = '사이즈')
    JOIN product_model_attribute pma3 ON rl.model_id = pma3.product_model_id
    JOIN attribute_value av3 ON pma3.attribute_value_id = av3.id AND av3.attribute_id = (SELECT id FROM attribute WHERE name = '소재')
"""

print("📦 recommend_log 데이터 조회 중...")
df = pd.read_sql(query, engine)

# timestamp 정규화
df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda x: int(x.timestamp()))
ts_min, ts_max = df['timestamp'].min(), df['timestamp'].max()
df['timestamp'] = (df['timestamp'] - ts_min) / (ts_max - ts_min + 1e-9)

# 저장
df.to_csv("recommend_log.csv", index=False, encoding='utf-8-sig')
print("✅ recommend_log.csv 저장 완료!")
