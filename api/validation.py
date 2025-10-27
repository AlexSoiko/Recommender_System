from flask import request, jsonify
from typing import Dict, Any, Tuple, Optional
import re


class RequestValidator:
    """Класс для валидации входящих запросов"""

    @staticmethod
    def validate_recommendation_request(data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Валидация запроса рекомендаций"""

        # Проверка обязательных полей
        required_fields = ['item_id']
        for field in required_fields:
            if field not in data:
                return False, {"error": f"Missing required field: {field}"}

        # Валидация item_id
        if not isinstance(data['item_id'], int) or data['item_id'] <= 0:
            return False, {"error": "item_id must be a positive integer"}

        # Валидация user_history
        if 'user_history' in data:
            if not isinstance(data['user_history'], list):
                return False, {"error": "user_history must be a list"}

            for item in data['user_history']:
                if not isinstance(item, int):
                    return False, {"error": "All items in user_history must be integers"}

        # Валидация n_recommendations
        if 'n_recommendations' in data:
            if not isinstance(data['n_recommendations'], int) or data['n_recommendations'] <= 0:
                return False, {"error": "n_recommendations must be a positive integer"}
            if data['n_recommendations'] > 20:  # Ограничение для производительности
                return False, {"error": "n_recommendations cannot exceed 20"}

        return True, None

    @staticmethod
    def validate_batch_request(data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Валидация пакетного запроса"""
        if 'items' not in data:
            return False, {"error": "Missing required field: items"}

        if not isinstance(data['items'], list):
            return False, {"error": "items must be a list"}

        if len(data['items']) > 50:  # Ограничение размера пакета
            return False, {"error": "Batch size cannot exceed 50 items"}

        for i, item_data in enumerate(data['items']):
            if not isinstance(item_data, dict):
                return False, {"error": f"Item at index {i} must be an object"}

            is_valid, error = RequestValidator.validate_recommendation_request(item_data)
            if not is_valid:
                error['item_index'] = i
                return False, error

        return True, None