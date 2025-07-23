import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ✅ 1. 정규화 대상 컬럼
timestamp_scaler = MinMaxScaler()

# ✅ 2. CSV 로딩
df = pd.read_csv("recommend_data_full.csv")

# ✅ 3. timestamp 정규화
df['timestamp_norm'] = timestamp_scaler.fit_transform(df[['timestamp']])
df.drop(columns=['timestamp'], inplace=True)

# ✅ 4. 범주형 컬럼 (Label Encoding용)
categorical_cols = ['gender', '색상', '사이즈', '소재']

# ✅ 5. Label Encoding
label_encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    label_encoders[col] = encoder  # 추후 디코딩용 저장

# ✅ 🚧 [선택사항] One-Hot Encoding 적용 시 사용
# onehot_df = pd.get_dummies(df, columns=categorical_cols)
# onehot_df.to_csv("recommend_data_onehot.csv", index=False, encoding="utf-8-sig")
# print("✅ One-Hot Encoding 적용 데이터 저장 완료 (recommend_data_onehot.csv)")

# ✅ 6. 저장
df.to_csv("recommend_data_encoded.csv", index=False, encoding="utf-8-sig")
print("✅ Label Encoding + 정규화 적용 데이터 저장 완료 (recommend_data_encoded.csv)")
