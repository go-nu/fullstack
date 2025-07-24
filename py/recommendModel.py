# recommendModel.py
import torch
import torch.nn as nn

class DeepRecModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=8):
        super().__init__()

        self.user_emb = nn.Embedding(num_embeddings['user_id'], embedding_dim)
        self.product_emb = nn.Embedding(num_embeddings['product_id'], embedding_dim)
        self.model_emb = nn.Embedding(num_embeddings['model_id'], embedding_dim)
        self.gender_emb = nn.Embedding(num_embeddings['gender'], 2)
        self.age_emb = nn.Embedding(num_embeddings['age_group'], 5)
        self.residence_emb = nn.Embedding(num_embeddings['residence_type'], 3)
        self.color_emb = nn.Embedding(num_embeddings['색상'], 4)
        self.size_emb = nn.Embedding(num_embeddings['사이즈'], 4)
        self.material_emb = nn.Embedding(num_embeddings['소재'], 4)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 2 + 5 + 3 + 4 + 4 + 4 + 1, 64),
            nn.ReLU(),
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
