import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def save_results_to_json(results: Dict[str, Any], filepath: str):
    """Сохранение результатов в JSON файл"""
    try:
        # Конвертация numpy типов в Python типы
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        results_serializable = convert_numpy_types(results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Результаты сохранены: {filepath}")

    except Exception as e:
        logger.error(f"Ошибка сохранения результатов: {e}")


def load_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Расчет важности признаков (для интерпретации модели)"""
    try:
        # Для нейронных сетей - используем веса первого слоя
        first_layer_weights = model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(first_layer_weights), axis=1)

        feature_importance = dict(zip(feature_names, importance))
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return sorted_importance

    except Exception as e:
        logger.warning(f"Не удалось рассчитать важность признаков: {e}")
        return {}