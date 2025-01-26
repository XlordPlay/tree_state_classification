import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

# Загрузка данных
df = pd.read_csv("/home/xlordplay/tree_state_classificator/tree_state_classification/data/2015-street-tree-census-tree-data.csv")

# Кодирование целевой переменной
df['health_encoded'] = df['health'].map({"Good": 2, "Fair": 1, "Poor": 0})

# Проверка на NaN в каждом столбце
print("Количество NaN в каждом столбце:")
print(df.isnull().sum())

# Удаление строк с NaN в целевой переменной
df = df.dropna(subset=['health_encoded'])

# Удаление строк с NaN в важных признаках
important_features = [
    'y_sp', 'x_sp', 'longitude', 'latitude', 'tree_dbh', 'census tract',
    'council district', 'trnk_other', 'brch_other', 'spc_latin'
]
df = df.dropna(subset=important_features)

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

# Проверка на NaN и бесконечные значения
print("Есть NaN в X:", np.isnan(X_scaled).any())
print("Есть бесконечные значения в X:", np.isinf(X_scaled).any())

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Преобразование данных в тензоры
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test.values)

# Определение функции инициализации весов
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Определение модели
class HealthModel(nn.Module):
    def __init__(self):
        super(HealthModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Инициализация модели, функции потерь и оптимизатора
model = HealthModel()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Инициализация весов
model.apply(initialize_weights)

# Обучение модели
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    
    loss = criterion(output, y_train_tensor)
    
    if torch.isnan(loss):
        print("Потеря стала nan на эпохе:", epoch + 1)
        break
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Оценка производительности модели
model.eval()
with torch.no_grad():
    output_test = model(X_test_tensor)
    _, predicted = torch.max(output_test, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Тестовая точность: {accuracy.item():.2f}")