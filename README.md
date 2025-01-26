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


Выбор архитектуры и построение DL-модели
1. Обоснование выбора архитектуры

Для задачи классификации состояния здоровья деревьев была выбрана архитектура глубокой нейронной сети с тремя полносвязными слоями. Вот подробности выбора:

    Входной слой: Размерность входного слоя соответствует количеству признаков после кодирования и нормализации. В данном случае это количество значений после применения OneHotEncoder и StandardScaler.

    Первый скрытый слой (128 нейронов): Используется для захвата сложных паттернов и взаимодействий в данных. Выбор 128 нейронов основан на том, что это достаточно большая емкость для обработки информации без значительного риска переобучения.

    Второй скрытый слой (64 нейрона): Этот слой уменьшает размерность, что помогает модели обобщать информацию и предотвращает переобучение. Он также продолжает выявлять более сложные паттерны.

    Выходной слой (3 нейрона): Соответствует количеству классов в целевой переменной (Good, Fair, Poor). Используется функция активации LogSoftmax, чтобы получить логарифмы вероятностей для каждого класса, что удобно для применения функции потерь NLLLoss.

    Функция активации ReLU: Эта функция активирует нейроны, что позволяет модели учиться быстрее и избегает проблем с затухающими градиентами, которые могут возникнуть при использовании сигмоидных функций.

2. Документация процесса принятия решений

    Выбор функций потерь и оптимизаторов: Использование NLLLoss в сочетании с LogSoftmax позволяет нам эффективно обрабатывать многоклассовую классификацию. Оптимизатор Adam был выбран за его эффективность и способность адаптивно регулировать скорость обучения.

    Инициализация весов: Инициализация весов с помощью метода Xavier позволяет избежать проблем с очень большими или очень маленькими значениями, что улучшает сходимость модели.

Оценка качества модели

Качество модели было оценено с использованием кросс-валидации с 5 фолдами, что обеспечило надежность результатов. Полученные метрики:

    Средняя точность: 0.81 ± 0.00: Это означает, что 81% предсказаний модели совпадают с фактическими значениями. Высокая точность указывает на хорошую работоспособность модели.

    Средняя F1-мера: 0.73 ± 0.00: F1-мера является гармоническим средним между точностью и полнотой. Значение 0.73 указывает на то, что модель хорошо справляется с классами, однако есть место для улучшения, особенно в отношении сбалансированности классов.

    Средняя полнота: 0.81 ± 0.00: Полнота показывает, насколько хорошо модель находит положительные примеры. Значение 0.81 говорит о том, что модель успешно идентифицирует большинство классов.

Заключение по архитектуре

В результате, предложенная архитектура и методы обучения обеспечивают надежную и эффективную классификацию состояния здоровья деревьев. Тем не менее, дальнейшие улучшения могут быть достигнуты путем оптимизации гиперпараметров, использования более сложных архитектур или применения методов увеличения данных.

Чтобы повысить уровень модели и улучшить её производительность, мы можем рассмотреть следующие стратегии:
1. Оптимизация гиперпараметров

    Скорость обучения: Попробуйте разные значения скорости обучения (например, 0.001, 0.0001) или используйте адаптивные методы, такие как ReduceLROnPlateau.
    Количество эпох: Увеличьте количество эпох, но следите за переобучением.
    Размер батча: Экспериментируйте с размером батча (например, 32, 64, 128).

2. Улучшение архитектуры

    Добавление слоев: Можно добавить больше скрытых слоев или увеличить количество нейронов в существующих слоях.
    Использование Dropout: Добавление Dropout слоев помогает предотвратить переобучение, отключая случайные нейроны во время обучения.
    Использование Batch Normalization: Это может помочь ускорить обучение и улучшить стабильность модели.

3. Улучшение предобработки данных

    Импутация пропусков: Используйте более сложные методы импутации для заполнения пропусков, например, KNN или регрессию.
    Аугментация данных: Если возможно, создавайте дополнительные данные путем аугментации для увеличения разнообразия обучающей выборки.
    Обработка выбросов: Проверьте данные на наличие выбросов и рассмотрите возможность их удаления или обработки.

4. Использование других алгоритмов

    Сложные модели: Попробуйте использовать более сложные архитектуры, такие как сверточные нейронные сети (CNN) или рекуррентные нейронные сети (RNN), если данные позволяют.
    Комбинирование моделей: Используйте ансамблирование (например, Random Forest, Gradient Boosting) для улучшения предсказательной способности.

5. Метрики и кросс-валидация

    Кросс-валидация: Убедитесь, что вы используете надежные методы кросс-валидации для оценки модели.
    Разнообразие метрик: Используйте дополнительные метрики (например, ROC AUC, точность, полноту) для более полной оценки производительности модели.

6. Обучение на большем количестве данных

    Дополнительные данные: Если возможно, собирайте больше данных для обучения модели.
    Синтетические данные: Используйте методы генерации синтетических данных для увеличения объема выборки.

7. Регуляризация

    L1 и L2 регуляризация: Добавьте регуляризацию к функции потерь, чтобы предотвратить переобучение.

