import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader

class RecommendDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # ✅ 특성 (입력)과 정답 (출력) 정의
        self.features = self.data[[
            'user_id', 'product_id', 'model_id',
            'gender', 'age_group', 'residence_type',
            '색상', '사이즈', '소재',
            'timestamp_norm'
        ]].values.astype(float)

        # ✅ label은 weight
        self.labels = self.data['weight'].values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y


# ✅ 데이터셋 로딩
dataset = RecommendDataset("recommend_data_encoded.csv")

# ✅ 데이터로더 생성
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)