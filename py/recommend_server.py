from fastapi import FastAPI
from pydantic import BaseModel
import torch
import mysql.connector
from recommendModel import DeepRecModel
import numpy as np

app = FastAPI()

num_embeddings = {
    'user_id': 3,
    'product_id': 4,
    'model_id': 12,
    'gender': 2,
    'age_group': 1,
    'residence_type': 3,
    '색상': 3,
    '사이즈': 3,
    '소재': 2,
}

# ✅ 모델 로딩
model = DeepRecModel(num_embeddings)
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model.eval()

def get_product_ids_from_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="woorizip"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id AS product_id FROM product")
    result = cursor.fetchall()
    cursor.close()
    conn.close()

    return [row[0] for row in result]

product_ids = get_product_ids_from_db()

# ✅ 사용자 프로필 입력 스키마
class UserProfile(BaseModel):
    user_id: int
    gender: int
    age_group: int
    residence_type: int
    색상: int
    사이즈: int
    소재: int

@app.post("/recommend")
def recommend(user: UserProfile):
    input_data = []
    for pid in product_ids:
        row = [
            user.user_id, pid, 0,
            user.gender,
            user.age_group,
            user.residence_type,
            user.색상,
            user.사이즈,
            user.소재,
            0.0
        ]
        input_data.append(row)

    # ✅ ndarray로 shape 안전하게 만들기
    X_np = np.array(input_data, dtype=np.float32)
    if X_np.ndim == 1:
        X_np = np.expand_dims(X_np, axis=0)

    X = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X).squeeze().numpy().tolist()

    results = sorted(zip(product_ids, scores), key=lambda x: -x[1])

    return [{"product_id": pid, "score": score} for pid, score in results]

# http://localhost:8000/docs#/