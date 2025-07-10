import mysql.connector
import pandas as pd
import numpy as np

# MySQL 연결 설정
conn = mysql.connector.connect(
    host="localhost",  # 호스트
    port=3306,         # 포트 (기본값: 3306)
    user="root",       # 사용자명
    password="1234",   # 비밀번호
    database="woorizip"
)

# 커서 생성
cursor = conn.cursor()

# 예시로 쿼리 실행
cursor.execute("SELECT * FROM recommend_log")

logs = cursor.fetchall()

# 연결 종료
cursor.close()
conn.close()

# DataFrame 생성
df = pd.DataFrame(logs, columns=['residence_type', 'id', 'product_id', 'timestamp', 'age_group', 'gender', 'nickname'])

# 1. age_group 처리
age_group_mapping = {'10s': 1, '20s': 2, '30s': 3, '40s': 4, '50over': 5}
df['age_group'] = df['age_group'].map(age_group_mapping)

# 2. gender 처리
gender_mapping = {'m': 0, 'f': 1}
df['gender'] = df['gender'].map(gender_mapping)

# 3. timestamp 처리 (유닉스 타임스탬프)
df['timestamp'] = df['timestamp'].apply(lambda x: int(x.timestamp()))

# 4. nickname 처리 (숫자 매핑)
df['nickname'] = df['nickname'].astype('category').cat.codes

# 결과 출력
print(df)

