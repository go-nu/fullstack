import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mysql.connector
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# ✅ 속성 개수를 DB에서 자동 조회하여 embedding_dim으로 사용
def get_embedding_dims_from_db(conn):
    query = """
    SELECT a.name AS attribute_name, COUNT(av.id) AS count
    FROM attribute a
    JOIN attribute_value av ON a.id = av.attribute_id
    GROUP BY a.name
    """
    df = pd.read_sql(query, conn)
    return {row['attribute_name']: row['count'] for _, row in df.iterrows()}

# ✅ DB에서 전체 학습 데이터 로딩 및 병합
def load_training_data(conn):
    # 1. 추천 로그
    log_df = pd.read_sql("""
        SELECT user_id, model_id, product_id, weight, timestamp
        FROM recommend_log
    """, conn)
    # 1. timestamp를 Unix 시간으로 변환
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp']).apply(lambda x: int(x.timestamp()))

    # 2. ✅ Min-Max 정규화
    ts_min = log_df['timestamp'].min()
    ts_max = log_df['timestamp'].max()
    log_df['timestamp'] = (log_df['timestamp'] - ts_min) / (ts_max - ts_min + 1e-9)  # 나눗셈 0 방지

    # 2. 사용자 정보
    user_df = pd.read_sql("""
        SELECT id AS user_id, gender, birth, residence_type
        FROM users
    """, conn)
    current_year = datetime.now().year
    user_df['age'] = current_year - pd.to_datetime(user_df['birth']).dt.year
    user_df['age_group'] = pd.cut(user_df['age'], bins=[-1,9,19,29,39,49,150], labels=[0,1,2,3,4,5]).astype(int)
    user_df.drop(['birth', 'age'], axis=1, inplace=True)

    # 3. 제품 정보
    product_df = pd.read_sql("SELECT id AS product_id, category_id FROM product", conn)

    # 4. 속성 정보 (색상, 사이즈, 소재)
    model_attr_df = pd.read_sql("""
        SELECT pma.product_model_id AS model_id, a.name AS attribute_type, av.value AS attribute_value
        FROM product_model_attribute pma
        JOIN attribute_value av ON pma.attribute_value_id = av.id
        JOIN attribute a ON av.attribute_id = a.id
    """, conn)

    # 속성 피벗 → 컬럼: 색상, 사이즈, 소재
    model_pivot = model_attr_df.pivot_table(
        index='model_id',
        columns='attribute_type',
        values='attribute_value',
        aggfunc='first'
    ).reset_index()

    # 병합
    df = log_df \
        .merge(user_df, on='user_id', how='left') \
        .merge(product_df, on='product_id', how='left') \
        .merge(model_pivot, on='model_id', how='left')

    # 모든 범주형 변수 정수 인코딩
    for col in ['user_id', 'product_id', 'model_id', 'gender', 'residence_type', '색상', '사이즈', '소재']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

# ✅ PyTorch Dataset 정의
class RecommendDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.features = self.data[[
            'user_id', 'product_id', 'model_id',
            'gender', 'age_group', 'residence_type',
            '색상', '사이즈', '소재', 'timestamp'
        ]].values.astype(float)
        self.labels = self.data['weight'].values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y

# ✅ 딥러닝 모델 정의
class DeepRecModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dims):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings['user_id'], 8)
        self.product_emb = nn.Embedding(num_embeddings['product_id'], 8)
        self.model_emb = nn.Embedding(num_embeddings['model_id'], 8)
        self.gender_emb = nn.Embedding(num_embeddings['gender'], 2)
        self.age_emb = nn.Embedding(num_embeddings['age_group'], 4)
        self.residence_emb = nn.Embedding(num_embeddings['residence_type'], 3)
        self.color_emb = nn.Embedding(num_embeddings['색상'], embedding_dims['색상'])
        self.size_emb = nn.Embedding(num_embeddings['사이즈'], embedding_dims['사이즈'])
        self.material_emb = nn.Embedding(num_embeddings['소재'], embedding_dims['소재'])

        total_input_dim = (
            8*3 + 2 + 4 + 3 +
            embedding_dims['색상'] + embedding_dims['사이즈'] + embedding_dims['소재'] + 1
        )

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        uid = x[:, 0].long()
        pid = x[:, 1].long()
        mid = x[:, 2].long()
        gender = x[:, 3].long()
        age = x[:, 4].long()
        res = x[:, 5].long()
        color = x[:, 6].long()
        size = x[:, 7].long()
        mat = x[:, 8].long()
        time = x[:, 9].unsqueeze(1)

        x = torch.cat([
            self.user_emb(uid), self.product_emb(pid), self.model_emb(mid),
            self.gender_emb(gender), self.age_emb(age), self.residence_emb(res),
            self.color_emb(color), self.size_emb(size), self.material_emb(mat),
            time
        ], dim=-1)

        return self.mlp(x).squeeze()

# ✅ 학습 시작
if __name__ == "__main__":
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="1234",
        database="woorizip"
    )

    # 자동 임베딩 크기 추출
    embedding_dims = get_embedding_dims_from_db(conn)
    df = load_training_data(conn)
    conn.close()

    dataset = RecommendDataset(df)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_embeddings = {col: df[col].nunique() for col in [
        'user_id', 'product_id', 'model_id',
        'gender', 'age_group', 'residence_type',
        '색상', '사이즈', '소재'
    ]}

    model = DeepRecModel(num_embeddings, embedding_dims)
    device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            sample_weight = y.clone()
            pred = model(X)

            # ⬇️ accuracy 계산
            predicted_labels = (pred > 0.5).float()
            correct = (predicted_labels == (y > 0).float()).sum().item()
            total_correct += correct
            total_samples += y.size(0)

            # loss 계산
            loss = criterion(pred, (y > 0).float())
            weighted_loss = (loss * sample_weight).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()

        acc = total_correct / total_samples if total_samples else 0
        print(f"Epoch {epoch + 1}/10 - Loss: {total_loss:.4f} - Accuracy: {acc:.4f}")

    # 필요 시 모델 저장
    # torch.save(model.state_dict(), "deeprec_model.pt")
