README: DL-модель для классификации состояния деревьев

Tree Health Classification Project
Описание проекта

Цель проекта — построить DL-модель для классификации состояния дерева (Good, Fair, Poor) по данным 2015 Street Tree Census (NYC).
Классификация поможет понять, какие факторы влияют на здоровье городских деревьев.
Данные

Датасет: 2015 Street Tree Census Tree Data (NYC).
Основные признаки:

    tree_dbh: Диаметр дерева на уровне груди (числовое значение).
    curb_loc: Расположение дерева относительно бордюра (OnCurb, OffsetFromCurb).
    steward: Признаки ухода за деревом (None/1or2/3or4/4orMore).
    guards: Тип защиты дерева (Harmful/Helpful/None/Unsure).
    sidewalk: Повреждение тротуара рядом с деревом (Damage/NoDamage).
    root_stone, root_grate, root_other: Проблемы с корнями (Yes/No).
    trunk_wire, trnk_light, trnk_other: Проблемы со стволом (Yes/No).
    brch_light, brch_shoe, brch_other: Проблемы с ветвями (Yes/No).
    spc_latin или spc_common: Латинское или общее название дерева.

Целевая переменная:

    health: Состояние дерева (Good/Fair/Poor).

Поля, которые удалены:
    
    tree_id, block_id, created_at: не содержат полезной информации для классификации.

    Географические поля (latitude, longitude, zipcode, boroname): исключены из-за низкой корреляции с состоянием деревьев.

    Административные поля (user_type, nta_name, и др.): не влияют на здоровье дерева.

Структура репозитория

    notebooks/eda.ipynb: Анализ и предобработка данных.
    src/train.py: Скрипт для обучения модели.
    src/inference.py: Скрипт для предсказаний.
    app/main.py: Простое API на FastAPI.
    README.md: Описание проекта и инструкции по запуску.

Используемые технологии

    Python
    PyTorch
    FastAPI
    Pandas, NumPy, Matplotlib, Seaborn (для анализа данных)

Запуск проекта

    Клонируйте репозиторий:

git clone https://github.com/ваш-репозиторий.git
cd ваш-репозиторий

Установите зависимости:

pip install -r requirements.txt

Проведите обучение:

python src/train.py

Запустите API:

    uvicorn app.main:app --reload

    Используйте эндпоинт /predict для предсказаний.

