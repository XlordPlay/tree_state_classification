from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import joblib
from sklearn.impute import SimpleImputer
import sys
import os

# Добавить src в sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from train_model import ImprovedHealthModel
# Загрузка модели и трансформеров
model = ImprovedHealthModel()
model.load_state_dict(torch.load("../tree_health_model.pth", weights_only=True))
model.eval()

column_transformer = joblib.load('column_transformer.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

# Определение модели для входных данных
class TreeData(BaseModel):
    y_sp: float
    x_sp: float
    longitude: float
    latitude: float
    tree_dbh: float
    census_tract: str
    council_district: str
    trnk_other: str
    brch_other: str
    spc_latin: str

@app.post("/predict/")
async def predict(tree_data: TreeData):
    # Преобразование входных данных в DataFrame
    input_data = pd.DataFrame([tree_data.dict()])

    # Заполнение отсутствующих значений
    imputer = SimpleImputer(strategy='most_frequent')
    input_data[['trnk_other', 'brch_other']] = imputer.fit_transform(input_data[['trnk_other', 'brch_other']])

    # Подготовка данных для инференса
    important_features = [
        'y_sp', 'x_sp', 'longitude', 'latitude', 'tree_dbh', 'census_tract',
        'council_district', 'trnk_other', 'brch_other', 'spc_latin'
    ]
    
    X_inference = input_data[important_features]

    # Кодирование и стандартизация
    X_encoded = column_transformer.transform(X_inference)
    X_scaled = scaler.transform(X_encoded)

    # Преобразование в тензоры
    X_tensor = torch.FloatTensor(X_scaled)

    # Выполнение предсказания
    with torch.no_grad():
        output = model(X_tensor)
        predicted_class = torch.argmax(output, dim=1).numpy()[0]

    # Маппинг классов
    class_mapping = {0: "Poor", 1: "Fair", 2: "Good"}
    predicted_health = class_mapping[predicted_class]

    return {"predicted_health": predicted_health}

# Чтобы запустить приложение, выполните команду:
# uvicorn app.main:app --reload