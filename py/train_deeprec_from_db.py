
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mysql.connector
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_training_data():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="1234",
        database="woorizip"
    )

    log_query = """
    SELECT user_id, model_id, product_id, weight, timestamp
    FROM recommend_log
    """
    log_df = pd.read_sql(log_query, conn)
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp']).apply(lambda x: int(x.timestamp()))

    user_query = """
    SELECT id AS user_id, gender, birth, residence_type
    FROM users
    """
    user_df = pd.read_sql(user_query, conn)
    current_year = datetime.now().year
    user_df['age'] = current_year - pd.to_datetime(user_df['birth']).dt.year
    user_df['age_group'] = pd.cut(
        user_df['age'],
        bins=[-1, 9, 19, 29, 39, 49, 150],
        labels=[0, 1, 2, 3, 4, 5],
        right=True
    ).astype(int)
    user_df.drop(['birth', 'age'], axis=1, inplace=True)

    product_query = """
    SELECT id AS product_id, category_id
    FROM product
    """
    product_df = pd.read_sql(product_query, conn)

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
    model_pivot = model_df.pivot_table(
        index='model_id',
        columns='attribute_type',
        values='attribute_value',
        aggfunc='first'
    ).reset_index()

    conn.close()

    merged = log_df         .merge(user_df, on='user_id', how='left')         .merge(product_df, on='product_id', how='left')         .merge(model_pivot, on='model_id', how='left')

    for col in ['user_id', 'product_id', 'model_id', 'gender', 'residence_type', '색상', '사이즈', '소재']:
        merged[col] = LabelEncoder().fit_transform(merged[col].astype(str))

    return merged

class RecommendDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.features = self.data[[
            'user_id', 'product_id', 'model_id',
            'gender', 'age_group', 'residence_type',
            '색상', '사이즈', '소재',
            'timestamp'
        ]].values.astype(float)
        self.labels = self.data['weight'].values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y

class DeepRecModel(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        emb_dim = 8
        self.user_emb = nn.Embedding(num_embeddings['user_id'], emb_dim)
        self.product_emb = nn.Embedding(num_embeddings['product_id'], emb_dim)
        self.model_emb = nn.Embedding(num_embeddings['model_id'], emb_dim)
        self.gender_emb = nn.Embedding(num_embeddings['gender'], 2)
        self.age_emb = nn.Embedding(num_embeddings['age_group'], 4)
        self.residence_emb = nn.Embedding(num_embeddings['residence_type'], 3)
        self.color_emb = nn.Embedding(num_embeddings['색상'], 4)
        self.size_emb = nn.Embedding(num_embeddings['사이즈'], 4)
        self.material_emb = nn.Embedding(num_embeddings['소재'], 4)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3 + 2 + 4 + 3 + 4 + 4 + 4 + 1, 64),
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
df = load_training_data()
dataset = RecommendDataset(df)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_embeddings = {col: df[col].nunique() for col in [
    'user_id', 'product_id', 'model_id',
    'gender', 'age_group', 'residence_type',
    '색상', '사이즈', '소재'
]}

model = DeepRecModel(num_embeddings)
device = torch.device("cpu")  # CPU 고정 실행
model = model.to(device)

criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        sample_weight = y.clone()
        pred = model(X)
        loss = criterion(pred, (y > 0).float())
        weighted_loss = (loss * sample_weight).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        epoch_loss += weighted_loss.item()
    print(f"✅ Epoch {epoch+1}/10 - Loss: {epoch_loss:.4f}")
