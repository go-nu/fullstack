import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_loader import load_training_data
import pandas as pd

# ✅ 1. Dataset 정의
class RecommendDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.copy()

        self.features = self.data[[  # 필요한 열만 추출
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

# 🔧 하이퍼파라미터 설정
embedding_dim = 8 # 4 or 8
hidden_size = 64
num_layers = 2
dropout_rate = 0.2
epochs = 10
batch_size = 32
learning_rate = 0.001

# ✅ 2. 모델 정의
class DeepRecModel(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()

        self.user_emb = nn.Embedding(num_embeddings['user_id'], embedding_dim)
        self.product_emb = nn.Embedding(num_embeddings['product_id'], embedding_dim)
        self.model_emb = nn.Embedding(num_embeddings['model_id'], embedding_dim)
        self.gender_emb = nn.Embedding(num_embeddings['gender'], 2)
        self.age_emb = nn.Embedding(num_embeddings['age_group'], 4)
        self.residence_emb = nn.Embedding(num_embeddings['residence_type'], 3)
        self.color_emb = nn.Embedding(num_embeddings['색상'], 4)
        self.size_emb = nn.Embedding(num_embeddings['사이즈'], 4)
        self.material_emb = nn.Embedding(num_embeddings['소재'], 4)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 2+4+3 + 4+4+4 + 1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
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

        user_vec = self.user_emb(uid)
        product_vec = self.product_emb(pid)
        model_vec = self.model_emb(mid)
        gender_vec = self.gender_emb(gender)
        age_vec = self.age_emb(age)
        res_vec = self.residence_emb(res)
        color_vec = self.color_emb(color)
        size_vec = self.size_emb(size)
        mat_vec = self.material_emb(mat)

        x = torch.cat([
            user_vec, product_vec, model_vec,
            gender_vec, age_vec, res_vec,
            color_vec, size_vec, mat_vec,
            time
        ], dim=-1)

        return self.mlp(x).squeeze()

# ✅ 3. 데이터 로딩
df = pd.read_csv("recommend_data_encoded.csv")
dataset = RecommendDataset("recommend_data_encoded.csv")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_embeddings = {
    'user_id': df['user_id'].nunique(),
    'product_id': df['product_id'].nunique(),
    'model_id': df['model_id'].nunique(),
    'gender': df['gender'].nunique(),
    'age_group': df['age_group'].nunique(),
    'residence_type': df['residence_type'].nunique(),
    '색상': df['색상'].nunique(),
    '사이즈': df['사이즈'].nunique(),
    '소재': df['소재'].nunique(),
}

model = DeepRecModel(num_embeddings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ✅ 4. 학습 루프
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
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

    print(f"✅ Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
