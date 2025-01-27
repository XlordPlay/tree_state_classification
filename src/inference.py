import pandas as pd
import torch
import joblib
from train_model import ImprovedHealthModel
from sklearn.impute import SimpleImputer

# Загрузка модели
model = ImprovedHealthModel()
model.load_state_dict(torch.load("/home/xlordplay/tree_state_classificator/tree_state_classification/tree_health_model.pth", weights_only=True))
model.eval()

# Загрузка трансформеров
column_transformer = joblib.load('column_transformer.pkl')
scaler = joblib.load('scaler.pkl')

# Загрузка данных
inference_df = pd.read_csv("/home/xlordplay/tree_state_classificator/tree_state_classification/data/inference_data.csv")

# Определение признаков
important_features = [
    'y_sp', 'x_sp', 'longitude', 'latitude', 'tree_dbh', 'census tract',
    'council district', 'trnk_other', 'brch_other', 'spc_latin'
]

# Проверка наличия признаков
missing_features = [feature for feature in important_features if feature not in inference_df.columns]
if missing_features:
    print(f"Отсутствующие признаки: {missing_features}")
else:
    # Создание имперта для обработки отсутствующих значений
    imputer = SimpleImputer(strategy='most_frequent')

    # Убедимся, что у нас есть ненулевые значения
    valid_features = [feature for feature in important_features if inference_df[feature].notnull().any()]
    if not valid_features:
        print("Все признаки имеют отсутствующие значения.")
    else:
        # Заполняем отсутствующие значения только для доступных признаков
        inference_df[valid_features] = imputer.fit_transform(inference_df[valid_features])

        # Заполнение проблемных столбцов вручную, если они все еще содержат NaN
        for feature in ['trnk_other', 'brch_other']:
            if feature in inference_df.columns:
                inference_df[feature].fillna('unknown', inplace=True)

    # Подготовка данных для инференса
    X_inference = inference_df[important_features]

    # Кодирование и стандартизация
    X_encoded = column_transformer.transform(X_inference)
    X_scaled = scaler.transform(X_encoded)

    # Преобразование в тензоры
    X_tensor = torch.FloatTensor(X_scaled)

    # Выполнение предсказания
    with torch.no_grad():
        output = model(X_tensor)
        predicted_classes = torch.argmax(output, dim=1)

    # Проверка длины предсказаний
    if len(predicted_classes) == len(inference_df):
        # Маппинг классов
        class_mapping = {0: "Poor", 1: "Fair", 2: "Good"}
        inference_df['predicted_health'] = predicted_classes.numpy()
        inference_df['predicted_health'] = inference_df['predicted_health'].map(class_mapping)

        # Сохранение результатов
        inference_df.to_csv("/home/xlordplay/tree_state_classificator/tree_state_classification/data/inference_results.csv", index=False)
        print(inference_df)
    else:
        print(f"Ошибка: длины не совпадают! Предсказания: {len(predicted_classes)}, Данные: {len(inference_df)}")