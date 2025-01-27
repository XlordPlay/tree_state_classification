import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
import joblib

# Загрузка данных
df = pd.read_csv("../data/2015-street-tree-census-tree-data.csv")

# Кодирование целевой переменной
df['health_encoded'] = df['health'].map({"Good": 2, "Fair": 1, "Poor": 0})

# Удаление строк с NaN
important_features = [
    'y_sp', 'x_sp', 'longitude', 'latitude', 'tree_dbh', 'census tract',
    'council district', 'trnk_other', 'brch_other', 'spc_latin'
]
df = df.dropna(subset=important_features + ['health_encoded'])

# Выбор признаков и целевой переменной
X = df[important_features]
y = df['health_encoded']

# Кодирование категориальных признаков
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), ['council district', 'trnk_other', 'brch_other', 'spc_latin']),
    ],
    remainder='passthrough'
)

X_encoded = column_transformer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Преобразование данных в тензоры
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.LongTensor(y.values)

# Определение модели
class ImprovedHealthModel(nn.Module):
    def __init__(self):
        super(ImprovedHealthModel, self).__init__()
        self.fc1 = nn.Linear(X_scaled.shape[1], 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.softmax(self.fc3(x))
        return x

# Инициализация модели, функции потерь и оптимизатора
model = ImprovedHealthModel()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Кросс-валидация
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
f1_scores = []
recalls = []

for train_index, test_index in kf.split(X_tensor):
    X_train_tensor, X_test_tensor = X_tensor[train_index], X_tensor[test_index]
    y_train_tensor, y_test_tensor = y_tensor[train_index], y_tensor[test_index]

    # Обучение модели
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Оценка производительности модели
    model.eval()
    with torch.no_grad():
        output_test = model(X_test_tensor)
        _, predicted = torch.max(output_test, 1)

        # Вычисление метрик
        accuracy = accuracy_score(y_test_tensor, predicted)
        f1 = f1_score(y_test_tensor, predicted, average='weighted')
        recall = recall_score(y_test_tensor, predicted, average='weighted')

        accuracies.append(accuracy)
        f1_scores.append(f1)
        recalls.append(recall)

# Вывод результатов
print(f"Средняя точность: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
print(f"Средняя F1-мера: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
print(f"Средняя полнота: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")

# Сохранение модели
torch.save(model.state_dict(), "tree_health_model.pth")
joblib.dump(column_transformer, 'column_transformer.pkl')
joblib.dump(scaler, 'scaler.pkl')