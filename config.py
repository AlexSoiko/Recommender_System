import os
from datetime import datetime

# Настройки путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Создание директорий
for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Настройки модели
MODEL_CONFIG = {
    'input_dim': None,  # Будет определено автоматически
    'hidden_layers': [128, 64, 32],
    'dropout_rates': [0.3, 0.3, 0.2],
    'activation': 'relu',
    'output_activation': 'softmax',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

# Настройки данных
DATA_CONFIG = {
    'test_size': 0.3,
    'val_size': 0.15,
    'random_state': 42
}

# Feature columns (будут определены автоматически)
FEATURE_COLUMNS = []