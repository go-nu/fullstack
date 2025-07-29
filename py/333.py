import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mysql.connector
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.metrics import roc_auc_score

# ✅ 시드 설정
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ✅ 속성 개수를 DB에서 자동 조회하여 embedding_dim으로 사용
def get_embedding_dims_from_db(conn):
    query = '''
            SELECT a.name AS attribute_name, COUNT(av.id) AS count
            FROM attribute a
                JOIN attribute_value av
            ON a.id = av.attribute_id
            GROUP BY a.name \
            '''
    df = pd.read_sql(query, conn)
    return {row['attribute_name']: row['count'] for _, row in df.iterrows()}

# ✅ DB에서 전체 학습 데이터 로딩 및 병합
def load_training_data(conn):
    log_df = pd.read_sql("""
                         SELECT user_id, model_id, product_id, weight, timestamp
                         FROM recommend_log
                         """, conn)
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp']).apply(lambda x: int(x.timestamp()))
    ts_min = log_df['timestamp'].min()
    ts_max = log_df['timestamp'].max()
    log_df['timestamp'] = (log_df['timestamp'] - ts_min) / (ts_max - ts_min + 1e-9)

    user_df = pd.read_sql("""
                          SELECT id AS user_id, gender, birth, residence_type
                          FROM users
                          """, conn)
    current_year = datetime.now().year
    user_df['age'] = current_year - pd.to_datetime(user_df['birth']).dt.year
    user_df['age_group'] = 1
    user_df.drop(['birth', 'age'], axis=1, inplace=True)

    product_df = pd.read_sql("SELECT id AS product_id, category_id FROM product", conn)

    model_attr_df = pd.read_sql("""
                                SELECT pma.product_model_id AS model_id,
                                       a.name               AS attribute_type,
                                       av.value             AS attribute_value
                                FROM product_model_attribute pma
                                         JOIN attribute_value av ON pma.attribute_value_id = av.id
                                         JOIN attribute a ON av.attribute_id = a.id
                                """, conn)
    model_pivot = model_attr_df.pivot_table(
        index='model_id',
        columns='attribute_type',
        values='attribute_value',
        aggfunc='first'
    ).reset_index()

    df = log_df.merge(user_df, on='user_id', how='left').merge(product_df, on='product_id', how='left').merge(
        model_pivot, on='model_id', how='left')

    df = df.dropna(subset=[
        'user_id', 'product_id', 'model_id',
        'gender', 'residence_type', 'age_group',
        '색상', '사이즈', '소재'
    ])

    for col in ['user_id', 'product_id', 'model_id',
                'gender', 'residence_type', '색상', '사이즈', '소재']:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])
    df['age_group'] = df['age_group'].astype(int)

    return df

# ✅ Dataset 클래스
class RecommendDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.copy()
        self.features = self.data[[
            'user_id', 'product_id', 'model_id',
            'gender', 'age_group', 'residence_type',
            '색상', '사이즈', '소재', 'timestamp', 'weight'
        ]].values.astype(float)
        self.labels = self.data['weight'].values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y

# ✅ 모델 클래스
class DeepRecModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dims):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings['user_id'] + 1, 4)
        self.product_emb = nn.Embedding(num_embeddings['product_id'] + 1, 4)
        self.model_emb = nn.Embedding(num_embeddings['model_id'] + 1, 4)
        self.gender_emb = nn.Embedding(num_embeddings['gender'] + 1, 2)
        self.age_emb = nn.Embedding(num_embeddings['age_group'] + 1, 1)
        self.residence_emb = nn.Embedding(num_embeddings['residence_type'] + 1, 3)
        self.color_emb = nn.Embedding(embedding_dims['색상'] + 1, embedding_dims['색상'])
        self.size_emb = nn.Embedding(embedding_dims['사이즈'] + 1, embedding_dims['사이즈'])
        self.material_emb = nn.Embedding(embedding_dims['소재'] + 1, embedding_dims['소재'])

        total_input_dim = 4*3 + 2 + 1 + 3 + \
                          embedding_dims['색상'] + embedding_dims['사이즈'] + embedding_dims['소재'] + 1 + 1

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
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
        weight_val = x[:, 10].unsqueeze(1)

        x = torch.cat([
            self.user_emb(uid), self.product_emb(pid), self.model_emb(mid),
            self.gender_emb(gender), self.age_emb(age), self.residence_emb(res),
            self.color_emb(color), self.size_emb(size), self.material_emb(mat),
            time, weight_val
        ], dim=-1)

        return self.mlp(x).squeeze()

# ✅ 학습 및 평가
if __name__ == "__main__":
    set_seed(42)
    conn = mysql.connector.connect(
        host="localhost", port=3306,
        user="root", password="1234",
        database="woorizip"
    )

    embedding_dims = get_embedding_dims_from_db(conn)
    df = load_training_data(conn)
    conn.close()

    user_ids = df['user_id'].unique()
    train_user_ids, val_user_ids = train_test_split(user_ids, test_size=0.2, random_state=42)
    train_df = df[df['user_id'].isin(train_user_ids)].reset_index(drop=True)
    val_df = df[df['user_id'].isin(val_user_ids)].reset_index(drop=True)

    train_dataset = RecommendDataset(train_df)
    val_dataset = RecommendDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_embeddings = {
        'user_id': int(df['user_id'].max()) + 1,
        'product_id': int(df['product_id'].max()) + 1,
        'model_id': int(df['model_id'].max()) + 1,
        'gender': int(df['gender'].max()) + 1,
        'age_group': int(df['age_group'].max()) + 1,
        'residence_type': int(df['residence_type'].max()) + 1,
        '색상': int(df['색상'].max()) + 1,
        '사이즈': int(df['사이즈'].max()) + 1,
        '소재': int(df['소재'].max()) + 1
    }

    model = DeepRecModel(num_embeddings, embedding_dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted_labels = (pred > 0.5).float()
            correct = (predicted_labels == (y > 0).float()).sum().item()
            total_correct += correct
            total_samples += y.size(0)
            loss = criterion(pred, (y > 0).float())
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = total_correct / total_samples if total_samples else 0
        print(f"[Train] Epoch {epoch + 1} - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                predicted_labels = (pred > 0.5).float()
                correct = (predicted_labels == (y > 0).float()).sum().item()
                val_correct += correct
                val_samples += y.size(0)
                loss = criterion(pred, (y > 0).float())
                loss = loss.mean()
                val_loss += loss.item()
        val_acc = val_correct / val_samples if val_samples else 0
        print(f"[Valid] Epoch {epoch + 1} - Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "deeprec_model.pt")

    def get_pytorch_precision_at_k(model, df, k=5):
        model.eval()
        user_ids = df['user_id'].unique()
        item_ids = df['model_id'].unique()
        precisions = []

        with torch.no_grad():
            for user_id in user_ids:
                user_data = df[df['user_id'] == user_id]
                seen_items = set(user_data['model_id'].values)
                candidate_items = [i for i in item_ids if i not in seen_items]

                if not candidate_items:
                    continue

                rows = []
                for item_id in candidate_items:
                    base = user_data.iloc[0].copy()
                    base['model_id'] = item_id
                    base['product_id'] = df[df['model_id'] == item_id]['product_id'].values[0]
                    base['weight'] = 0
                    rows.append(base)

                input_df = pd.DataFrame(rows)
                X = input_df[[
                    'user_id', 'product_id', 'model_id', 'gender', 'age_group',
                    'residence_type', '색상', '사이즈', '소재', 'timestamp', 'weight'
                ]].values
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                preds = model(X_tensor).cpu().numpy()

                top_k_idx = preds.argsort()[::-1][:k]
                top_k_items = input_df.iloc[top_k_idx]['model_id'].values
                relevant_items = set(df[(df['user_id'] == user_id) & (df['weight'] > 0)]['model_id'])

                hit = sum([1 for item in top_k_items if item in relevant_items])
                precisions.append(hit / k)

        return sum(precisions) / len(precisions)

    def get_pytorch_auc(model, dataset):
        model.eval()
        y_true, y_score = [], []

        with torch.no_grad():
            for X, y in DataLoader(dataset, batch_size=64):
                X, y = X.to(device), y.to(device)
                pred = model(X).cpu().numpy()
                y_true.extend((y == 3).int().cpu().numpy())
                y_score.extend(pred)

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        if len(np.unique(y_true)) < 2:
            print("⚠️ AUC 계산 불가: y_true에 클래스가 하나뿐입니다.")
            return 0.0

        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception as e:
            print(f"⚠️ AUC 계산 중 오류: {e}")
            auc = 0.0

        return auc

    pt_precision = get_pytorch_precision_at_k(model, df, k=5)
    pt_auc = get_pytorch_auc(model, RecommendDataset(df))
    print(f"📊 Precision@5: {pt_precision:.4f}")
    print(f"📊 AUC Score: {pt_auc:.4f}")
