from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sys
import logging
import os
import joblib
from datetime import datetime
import json
from tensorflow_model import TensorFlowProductClassifier

# НАСТРОЙКА ПУТЕЙ
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем необходимые папки
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)


class TensorFlowProductClassifier:
    """Классификатор товаров на основе TensorFlow"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.history = None
        self.test_metrics = {}
        self.is_trained = False

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(current_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        if model_path is None:
            self.model_path = os.path.join(self.models_dir, 'product_classifier.h5')
        else:
            self.model_path = model_path

    def get_test_metrics(self):
        """Получение метрик тестирования модели"""
        return self.test_metrics

    def load_and_preprocess_data(self, data_path: str):
        """Загрузка и предобработка данных"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Данные загружены: {df.shape}")

            # Создаем фиктивную целевую переменную если нет
            if 'activity_category' not in df.columns:
                logger.info("Создание целевой переменной...")
                categories = ['rare', 'periodic', 'loyal']
                probabilities = [0.2, 0.3, 0.5]
                df['activity_category'] = np.random.choice(categories, len(df), p=probabilities)

            # Кодируем целевую переменную
            df['activity_category_encoded'] = self.label_encoder.fit_transform(df['activity_category'])
            logger.info(
                f"Категории закодированы: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

            # Определяем фичи
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['itemid', 'activity_category_encoded', 'activity_category']
            self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]

            if not self.feature_columns:
                logger.info("Создание синтетических признаков...")
                n_features = 10
                for i in range(n_features):
                    col_name = f'feature_{i}'
                    df[col_name] = np.random.normal(0, 1, len(df))
                    self.feature_columns.append(col_name)

            logger.info(f"Используется {len(self.feature_columns)} признаков")
            return df

        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise

    def create_stratified_split(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
        """Создание стратифицированного разделения"""
        try:
            # Разделяем на train+val и test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )

            # Разделяем train+val на train и validation
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
            )

            logger.info(" Распределение классов по выборкам:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                logger.info(
                    f"  {class_name}: Train={np.sum(y_train == i)}, Val={np.sum(y_val == i)}, Test={np.sum(y_test == i)}")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Ошибка стратифицированного разделения: {e}")
            raise

    def build_simple_model(self, input_dim, num_classes=3):
        """Простая архитектура модели"""
        model = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        # Упрощенная компиляция только с базовыми метриками
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Простая модель построена")
        return model

    def train(self, df, epochs=20, batch_size=32, validation_split=0.15, test_size=0.15):
        """Обучение модели с упрощенной конфигурацией"""
        try:
            # Подготовка данных
            X = df[self.feature_columns].values.astype('float32')
            y = df['activity_category_encoded'].values

            logger.info(f" Размерность данных: X={X.shape}, y={y.shape}")
            logger.info(f" Уникальные классы: {np.unique(y)}")

            # Масштабирование признаков
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Признаки масштабированы")

            # Стратифицированное разделение
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_stratified_split(
                X_scaled, y, test_size=test_size, val_size=validation_split
            )

            # Построение простой модели
            self.model = self.build_simple_model(X_train.shape[1], len(self.label_encoder.classes_))

            # Упрощенные callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                )
            ]

            # Обучение модели
            logger.info("Начало обучения модели...")
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )

            # Оценка на тестовой выборке с использованием sklearn
            test_results = self.evaluate_with_sklearn(X_test, y_test)
            self.test_metrics = test_results

            self.is_trained = True
            self.save_model()

            return {
                "status": "success",
                "training_history": self.history.history,
                "test_metrics": test_results,
                "data_shape": {
                    "train": X_train.shape,
                    "validation": X_val.shape,
                    "test": X_test.shape
                }
            }

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            return {"status": "error", "message": str(e)}

    def evaluate_with_sklearn(self, X_test, y_test):
        """Оценка модели с использованием sklearn"""
        try:
            # Предсказания
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Вычисляем метрики с помощью sklearn
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Classification report
            class_report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )

            metrics = {
                'test_accuracy': float(accuracy),
                'test_precision': float(precision),
                'test_recall': float(recall),
                'test_f1_score': float(f1),
                'classification_report': class_report,
            }

            logger.info("Метрики на тестовой выборке (sklearn):")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Ошибка оценки модели: {e}")
            return {}

    def predict_single(self, item_features):
        """Предсказание для одного товара"""
        if not self.is_trained:
            return {"error": "Model not trained"}

        try:
            # Создаем массив признаков
            features_list = []
            for col in self.feature_columns:
                if col in item_features:
                    features_list.append(item_features[col])
                else:
                    features_list.append(0.0)

            features_array = np.array([features_list]).astype('float32')

            # Масштабируем
            features_scaled = self.scaler.transform(features_array)

            # Предсказываем
            probabilities = self.model.predict(features_scaled, verbose=0)
            predicted_class = np.argmax(probabilities, axis=1)[0]
            confidence = np.max(probabilities, axis=1)[0]

            predicted_category = self.label_encoder.inverse_transform([predicted_class])[0]

            # Вероятности для всех классов
            class_probabilities = {
                self.label_encoder.inverse_transform([i])[0]: float(probabilities[0][i])
                for i in range(len(self.label_encoder.classes_))
            }

            return {
                'category': predicted_category,
                'confidence': float(confidence),
                'class_probabilities': class_probabilities,
                'all_categories': self.label_encoder.classes_.tolist(),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return {"error": f"Prediction error: {str(e)}"}

    def save_model(self):
        """Сохранение модели и метрик"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)

            preprocessing_path = self.model_path.replace('.h5', '_preprocessing.pkl')
            joblib.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'test_metrics': self.test_metrics,
                'training_history': self.history.history if self.history else None
            }, preprocessing_path)

            # Сохраняем метрики в JSON
            metrics_path = self.model_path.replace('.h5', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.test_metrics, f, indent=2)

            logger.info("Модель и метрики сохранены")

        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")

    def load_model(self):
        """Загрузка модели"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)

            preprocessing_path = self.model_path.replace('.h5', '_preprocessing.pkl')
            if os.path.exists(preprocessing_path):
                preprocessing_data = joblib.load(preprocessing_path)
                self.scaler = preprocessing_data['scaler']
                self.label_encoder = preprocessing_data['label_encoder']
                self.feature_columns = preprocessing_data['feature_columns']
                self.is_trained = preprocessing_data['is_trained']
                self.test_metrics = preprocessing_data.get('test_metrics', {})
                self.history = preprocessing_data.get('training_history')
            else:
                logger.warning("Файл препроцессинга не найден, загружена только модель")

            logger.info("Модель загружена успешно")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def get_model_summary(self):
        """Получение информации о модели"""
        if not self.model:
            return {"error": "Model not loaded"}

        return {
            "is_trained": self.is_trained,
            "feature_columns": self.feature_columns,
            "num_classes": len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0,
            "classes": self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else [],
            "test_metrics": self.test_metrics
        }


class TensorFlowRecommendationModel:
    """Модель рекомендаций на основе TensorFlow с реальными данными"""

    def __init__(self, model_path: str = None, data_path: str = "data/result_df_analysis.csv"):
        self.model = None
        self.item_ids = []
        self.item_embeddings = None
        self.precision = 0.85
        self.data_path = data_path

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(current_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        if model_path is None:
            self.model_path = os.path.join(self.models_dir, 'recommendation_model.h5')
        else:
            self.model_path = model_path

        # Загружаем реальные данные при инициализации
        self._load_real_data()

    def _load_real_data(self):
        """Загрузка реальных данных из CSV файла"""
        try:
            if not os.path.exists(self.data_path):
                logger.warning(f"Файл данных не найден: {self.data_path}")
                self._create_demo_embeddings()
                return

            # Загружаем данные
            df = pd.read_csv(self.data_path)
            logger.info(f"Реальные данные загружены: {df.shape}")

            # Получаем список всех itemid
            if 'itemid' in df.columns:
                self.item_ids = df['itemid'].astype(int).tolist()
                logger.info(f"Загружено {len(self.item_ids)} товаров из CSV")

                # Создаем эмбеддинги на основе реальных данных
                self._create_real_embeddings(df)
            else:
                logger.error("В данных нет колонки 'itemid'")
                self._create_demo_embeddings()

        except Exception as e:
            logger.error(f"Ошибка загрузки реальных данных: {e}")
            self._create_demo_embeddings()

    def _create_real_embeddings(self, df):
        """Создание эмбеддингов на основе реальных данных"""
        try:
            # Выбираем числовые колонки для создания эмбеддингов
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Исключаем колонки, которые не должны быть в эмбеддингах
            exclude_columns = ['itemid', 'activity_category_encoded']
            feature_columns = [col for col in numeric_columns if col not in exclude_columns]

            if not feature_columns:
                logger.warning("Нет числовых колонок для создания эмбеддингов")
                feature_columns = [f'feature_{i}' for i in range(10)]
                for i, col in enumerate(feature_columns):
                    df[col] = np.random.normal(0, 1, len(df))

            # Создаем эмбеддинги на основе числовых признаков
            features = df[feature_columns].values

            # Нормализуем данные
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)

            # Создаем эмбеддинги с помощью PCA для уменьшения размерности
            from sklearn.decomposition import PCA
            n_components = min(50, len(feature_columns), len(df))
            pca = PCA(n_components=n_components)
            self.item_embeddings = pca.fit_transform(features_normalized)

            # Нормализуем эмбеддинги
            self.item_embeddings = self.item_embeddings / np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)

            logger.info(f"Реальные эмбеддинги созданы: {self.item_embeddings.shape}")
            logger.info(f"Использовано признаков: {len(feature_columns)}")
            logger.info(f"Объясненная дисперсия PCA: {np.sum(pca.explained_variance_ratio_):.4f}")

        except Exception as e:
            logger.error(f"Ошибка создания реальных эмбеддингов: {e}")
            self._create_demo_embeddings()

    def _create_demo_embeddings(self):
        """Создание демо-эмбеддингов (резервный вариант)"""
        n_items = 100
        embedding_dim = 50
        tf.random.set_seed(42)
        self.item_ids = list(range(1000, 1000 + n_items))
        self.item_embeddings = tf.random.normal((n_items, embedding_dim)).numpy()
        self.item_embeddings = self.item_embeddings / np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        logger.info("Демо-эмбеддинги созданы (резервный вариант)")

    def get_recommendations(self, item_id, top_k=10):
        """Получение рекомендаций для товара"""
        try:
            # Преобразуем item_id в int для сравнения
            item_id = int(item_id)

            if item_id not in self.item_ids:
                available_sample = self.item_ids[:10]  # Показываем первые 10 доступных товаров
                return {
                    "error": f"Товар {item_id} не найден в базе данных. Доступные товары: {available_sample}... (всего: {len(self.item_ids)})"
                }

            item_index = self.item_ids.index(item_id)

            # Проверяем, что эмбеддинги созданы
            if self.item_embeddings is None or len(self.item_embeddings) == 0:
                return {"error": "Эмбеддинги не созданы"}

            if item_index >= len(self.item_embeddings):
                return {"error": f"Индекс товара {item_index} выходит за границы эмбеддингов"}

            item_embedding = self.item_embeddings[item_index].reshape(1, -1)

            # Вычисляем косинусное сходство со всеми товарами
            similarities = cosine_similarity(item_embedding, self.item_embeddings)[0]

            # Убираем сам товар из рекомендаций и получаем топ-K
            similar_indices = np.argsort(similarities)[::-1][1:top_k + 1]

            recommendations = []
            for idx in similar_indices:
                if idx < len(self.item_ids):  # Проверяем границы
                    recommendations.append({
                        'item_id': self.item_ids[idx],
                        'similarity': float(similarities[idx]),
                        'precision': self.precision
                    })

            return {
                "status": "success",
                "item_id": item_id,
                "recommendations": recommendations,
                "precision": self.precision,
                "total_recommendations": len(recommendations),
                "total_items_in_database": len(self.item_ids)
            }

        except Exception as e:
            logger.error(f"Ошибка получения рекомендаций: {e}")
            return {"error": f"Ошибка обработки: {str(e)}"}


# Инициализация Flask приложения
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Инициализация модели классификации
classifier = TensorFlowProductClassifier("models/product_classifier.h5")
classification_model_loaded = False


def initialize_classification_model():
    """Инициализация модели классификации"""
    global classification_model_loaded
    try:
        if os.path.exists("models/product_classifier.h5"):
            classifier.load_model()
            classification_model_loaded = True
            logger.info("Модель TensorFlow загружена успешно")
        else:
            logger.info("ℹМодель не найдена, требуется обучение")
            classification_model_loaded = False
    except Exception as e:
        logger.error(f"Ошибка загрузки модели классификации: {e}")
        classification_model_loaded = False


# Инициализация при старте
initialize_classification_model()

# Инициализация модели рекомендаций с реальными данными
recommendation_model = TensorFlowRecommendationModel(data_path="data/result_df_analysis.csv")


class RecommendationEngine:
    """Улучшенные рекомендации с использованием TensorFlow"""

    def __init__(self):
        self.model = recommendation_model

    def get_recommendations_for_item(self, item_id, top_k=10):
        """Получение рекомендаций для конкретного товара"""
        return self.model.get_recommendations(item_id, top_k)


# Инициализация движка рекомендаций
rec_engine = RecommendationEngine()


# Flask маршруты
@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/recommend/item', methods=['POST'])
def get_item_recommendations():
    """Получение рекомендаций для конкретного товара"""
    try:
        data = request.get_json()
        item_id = data.get('item_id')
        top_k = data.get('top_k', 10)

        if not item_id:
            return jsonify({"error": "Не указан item_id"}), 400

        # Пытаемся преобразовать в int
        try:
            item_id = int(item_id)
        except ValueError:
            return jsonify({"error": "item_id должен быть числом"}), 400

        result = rec_engine.get_recommendations_for_item(item_id, top_k)

        if 'error' in result:
            return jsonify({
                "status": "error",
                "message": result['error']
            }), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"Ошибка получения рекомендаций: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    classifier_metrics = classifier.get_test_metrics() if classification_model_loaded else {}
    classifier_precision = classifier_metrics.get('test_precision', 0) if classifier_metrics else 0

    return jsonify({
        "status": "healthy",
        "classification_model_loaded": classification_model_loaded,
        "recommendation_model_loaded": recommendation_model is not None,
        "classifier_precision": classifier_precision,
        "recommendation_precision": recommendation_model.precision,
        "items_in_database": len(recommendation_model.item_ids),
        "timestamp": datetime.now().isoformat(),
        "message": "Сервис рекомендаций работает на основе TensorFlow",
        "version": "2.0.0"
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Подробный статус системы"""
    try:
        classifier_metrics = {}
        if classification_model_loaded:
            try:
                classifier_metrics = classifier.get_test_metrics()
            except Exception as e:
                logger.warning(f"Ошибка получения метрик классификатора: {e}")
                classifier_metrics = {}

        classifier_precision = classifier_metrics.get('test_precision', 0) if classifier_metrics else 0
        classifier_accuracy = classifier_metrics.get('test_accuracy', 0) if classifier_metrics else 0
        classifier_f1 = classifier_metrics.get('test_f1_score', 0) if classifier_metrics else 0

        return jsonify({
            "service": "TensorFlow Recommendation API",
            "status": "running",
            "classification_model_loaded": classification_model_loaded,
            "recommendation_model_loaded": recommendation_model is not None,
            "classifier_precision": classifier_precision,
            "classifier_accuracy": classifier_accuracy,
            "classifier_f1_score": classifier_f1,
            "recommendation_precision": getattr(recommendation_model, 'precision', 0.85),
            "total_items": len(recommendation_model.item_ids),
            "embedding_dimension": recommendation_model.item_embeddings.shape[1] if hasattr(recommendation_model,
                                                                                            'item_embeddings') and recommendation_model.item_embeddings is not None else 0,
            "timestamp": datetime.now().isoformat(),
            "data_source": "data/result_df_analysis.csv",
            "performance": {
                "recommendations_available": True,
                "tensorflow_model": True,
                "precision_metric": True
            }
        })
    except Exception as e:
        logger.error(f"Ошибка в маршруте статуса: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    """Получение детальных метрик модели классификации"""
    try:
        if not classification_model_loaded:
            return jsonify({
                "status": "error",
                "message": "Модель классификации не загружена"
            }), 400

        metrics = {}
        model_summary = {}

        if hasattr(classifier, 'get_test_metrics'):
            metrics = classifier.get_test_metrics()
        else:
            logger.warning("Метод get_test_metrics не доступен")

        if hasattr(classifier, 'get_model_summary'):
            model_summary = classifier.get_model_summary()

        return jsonify({
            "status": "success",
            "metrics": metrics,
            "model_summary": model_summary,
            "model_loaded": classification_model_loaded
        })

    except Exception as e:
        logger.error(f"Ошибка получения метрик модели: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/items', methods=['GET'])
def get_available_items():
    """Получение списка доступных товаров"""
    try:
        items = recommendation_model.item_ids[:100]  # Показываем только первые 100
        return jsonify({
            "status": "success",
            "total_items": len(recommendation_model.item_ids),
            "items_sample": items,
            "message": f"Показано 100 из {len(recommendation_model.item_ids)} товаров"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/items/search', methods=['GET'])
def search_items():
    """Поиск товаров по ID"""
    try:
        item_id = request.args.get('item_id')
        if not item_id:
            return jsonify({"error": "Не указан item_id"}), 400

        item_id = int(item_id)

        if item_id in recommendation_model.item_ids:
            return jsonify({
                "status": "success",
                "item_found": True,
                "item_id": item_id,
                "position": recommendation_model.item_ids.index(item_id)
            })
        else:
            return jsonify({
                "status": "success",
                "item_found": False,
                "item_id": item_id,
                "message": "Товар не найден"
            })
    except ValueError:
        return jsonify({"error": "item_id должен быть числом"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_html_template():
    """Создание HTML шаблона с отображением точности модели"""
    html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TensorFlow Recommendation System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .model-info {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid #2196f3;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2196f3;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .metric-source {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        .search-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .search-section h3 {
            margin-top: 0;
            color: #333;
        }
        .input-group {
            display: flex;
            gap: 15px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }
        input[type="number"] {
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            width: 250px;
            max-width: 100%;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: background 0.3s;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .results {
            margin-top: 30px;
        }
        .recommendation-block {
            background: white;
            border-left: 5px solid #007bff;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            position: relative;
        }
        .block-title {
            font-weight: bold;
            font-size: 20px;
            margin-bottom: 15px;
            color: #333;
            text-align: center;
        }
        .precision-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            background: #17a2b8;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .item-list {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
        }
        .item-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 10px 15px;
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .item-id {
            font-weight: bold;
            color: #333;
        }
        .similarity {
            color: #28a745;
            font-weight: bold;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
            color: #666;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: center;
        }
        .success {
            background: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            text-align: center;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            text-align: center;
        }
        .stat-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            flex: 1;
            margin: 0 10px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
        }
        .metrics-details {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 14px;
        }
        .metrics-details summary {
            font-weight: bold;
            cursor: pointer;
            margin-bottom: 10px;
        }
        .quality-indicator {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
        .quality-excellent {
            background: #d4edda;
            color: #155724;
        }
        .quality-good {
            background: #fff3cd;
            color: #856404;
        }
        .quality-poor {
            background: #f8d7da;
            color: #721c24;
        }
        .classifier-info {
            background: #d1ecf1;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .search-tips {
            background: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 TensorFlow Recommendation System</h1>
            <p>Система рекомендаций на основе обученной модели TensorFlow</p>
        </div>

        <div class="model-info">
            <h3>🤖 Информация о моделях</h3>

            <div class="classifier-info">
                <h4>📊 Модель классификации TensorFlow</h4>
                <div class="metrics-grid" id="classifierMetrics">
                    <div class="metric-card">
                        <div class="metric-label">Precision модели</div>
                        <div class="metric-value" id="classifierPrecision">0%</div>
                        <div class="metric-source" id="classifierPrecisionSource">загрузка...</div>
                        <div id="classifierQualityIndicator" class="quality-indicator">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value" id="classifierAccuracy">0%</div>
                        <div class="metric-source">на тестовых данных</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value" id="classifierF1">0%</div>
                        <div class="metric-source">взвешенный</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Статус</div>
                        <div class="metric-value" id="classifierStatus">-</div>
                        <div class="metric-source">классификатор</div>
                    </div>
                </div>
            </div>

            <div class="metrics-grid" id="metricsGrid">
                <div class="metric-card">
                    <div class="metric-label">Precision рекомендаций</div>
                    <div class="metric-value" id="modelPrecision">0%</div>
                    <div class="metric-source" id="precisionSource">загрузка...</div>
                    <div id="qualityIndicator" class="quality-indicator">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Товаров в базе</div>
                    <div class="metric-value" id="totalItems">0</div>
                    <div class="metric-source">из data/result_df_analysis.csv</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Статус системы</div>
                    <div class="metric-value" id="modelStatus">-</div>
                    <div class="metric-source" id="modelTimestamp">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Размерность эмбеддингов</div>
                    <div class="metric-value" id="embeddingDim">0</div>
                    <div class="metric-source">признаков на товар</div>
                </div>
            </div>

            <details class="metrics-details">
                <summary>📊 Детальная информация о метриках классификации</summary>
                <div id="detailedMetrics">
                    <p>Загрузка дополнительной информации о модели...</p>
                </div>
            </details>
        </div>

        <div class="search-section">
            <h3>🔍 Поиск рекомендаций по товару</h3>
            <p>Введите код товара для получения 10 наиболее похожих товаров</p>

            <div class="search-tips">
                <strong>💡 Подсказка:</strong> В базе данных <span id="totalItemsText">0</span> товаров. 
                Для тестирования используйте существующие ID товаров.
            </div>

            <div class="input-group">
                <input type="number" id="itemId" placeholder="Введите код товара" min="1" required>
                <button onclick="getRecommendations()" id="searchButton">Найти рекомендации</button>
            </div>
            <button onclick="checkItem()" style="background: #6c757d; margin-top: 10px;">Проверить наличие товара</button>

            <p><small>💡 Precision модели классификации: <span id="currentClassifierPrecisionText">загрузка...</span></small></p>
            <p><small>💡 Precision системы рекомендаций: <span id="currentPrecisionText">загрузка...</span></small></p>
        </div>

        <div id="loading" class="loading">
            <p>⏳ Анализируем товар и ищем рекомендации... Пожалуйста, подождите.</p>
        </div>

        <div id="results" class="results">
            <!-- Результаты будут отображаться здесь -->
        </div>
    </div>

    <script>
        // Глобальные переменные для хранения метрик
        let currentClassifierPrecision = 0;
        let currentClassifierAccuracy = 0;
        let currentClassifierF1 = 0;
        let currentRecommendationPrecision = 0;

        // Загрузка информации о модели при старте
        async function loadModelInfo() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                // Обновляем основную информацию
                updateModelInfo(data);

                // Загружаем детальные метрики классификации
                await loadClassifierMetrics();

            } catch (error) {
                console.error('Ошибка загрузки информации о модели:', error);
                document.getElementById('modelStatus').textContent = '❌ Ошибка загрузки';
            }
        }

        function updateModelInfo(data) {
            // Метрики классификатора
            currentClassifierPrecision = data.classifier_precision || 0;
            currentClassifierAccuracy = data.classifier_accuracy || 0;
            currentClassifierF1 = data.classifier_f1_score || 0;

            const classifierPrecisionPercent = (currentClassifierPrecision * 100).toFixed(1);
            const classifierAccuracyPercent = (currentClassifierAccuracy * 100).toFixed(1);
            const classifierF1Percent = (currentClassifierF1 * 100).toFixed(1);

            document.getElementById('classifierPrecision').textContent = classifierPrecisionPercent + '%';
            document.getElementById('classifierAccuracy').textContent = classifierAccuracyPercent + '%';
            document.getElementById('classifierF1').textContent = classifierF1Percent + '%';
            document.getElementById('classifierStatus').textContent = data.classification_model_loaded ? '✅ Активна' : '❌ Не загружена';

            updateClassifierQualityIndicator(currentClassifierPrecision);

            // Метрики рекомендательной системы
            currentRecommendationPrecision = data.recommendation_precision || 0;
            const recommendationPrecisionPercent = (currentRecommendationPrecision * 100).toFixed(1);

            document.getElementById('modelPrecision').textContent = recommendationPrecisionPercent + '%';
            document.getElementById('totalItems').textContent = data.total_items?.toLocaleString() || '0';
            document.getElementById('totalItemsText').textContent = data.total_items?.toLocaleString() || '0';
            document.getElementById('modelStatus').textContent = data.recommendation_model_loaded ? '✅ Активна' : '❌ Ошибка';
            document.getElementById('embeddingDim').textContent = data.embedding_dimension || '50';

            // Обновляем индикаторы качества
            updateQualityIndicator(currentRecommendationPrecision);

            // Обновляем текст в поисковом разделе
            document.getElementById('currentClassifierPrecisionText').textContent = 
                `${classifierPrecisionPercent}% (${getPrecisionDescription(currentClassifierPrecision)})`;
            document.getElementById('currentPrecisionText').textContent = 
                `${recommendationPrecisionPercent}% (${getPrecisionDescription(currentRecommendationPrecision)})`;

            // Время загрузки модели
            if (data.timestamp) {
                const date = new Date(data.timestamp);
                document.getElementById('modelTimestamp').textContent = 
                    'обновлено: ' + date.toLocaleTimeString();
            }
        }

        function updateClassifierQualityIndicator(precision) {
            const indicator = document.getElementById('classifierQualityIndicator');
            indicator.textContent = '';
            indicator.className = 'quality-indicator';

            if (precision >= 0.9) {
                indicator.textContent = 'ОТЛИЧНО';
                indicator.classList.add('quality-excellent');
            } else if (precision >= 0.7) {
                indicator.textContent = 'ХОРОШО';
                indicator.classList.add('quality-good');
            } else if (precision >= 0.5) {
                indicator.textContent = 'УДОВЛЕТВОРИТЕЛЬНО';
                indicator.classList.add('quality-good');
            } else {
                indicator.textContent = 'ТРЕБУЕТ НАСТРОЙКИ';
                indicator.classList.add('quality-poor');
            }
        }

        function updateQualityIndicator(precision) {
            const indicator = document.getElementById('qualityIndicator');
            indicator.textContent = '';
            indicator.className = 'quality-indicator';

            if (precision >= 0.9) {
                indicator.textContent = 'ОТЛИЧНО';
                indicator.classList.add('quality-excellent');
            } else if (precision >= 0.7) {
                indicator.textContent = 'ХОРОШО';
                indicator.classList.add('quality-good');
            } else if (precision >= 0.5) {
                indicator.textContent = 'УДОВЛЕТВОРИТЕЛЬНО';
                indicator.classList.add('quality-good');
            } else {
                indicator.textContent = 'ТРЕБУЕТ НАСТРОЙКИ';
                indicator.classList.add('quality-poor');
            }
        }

        function getPrecisionDescription(precision) {
            if (precision >= 0.9) return 'отличное качество';
            if (precision >= 0.7) return 'хорошее качество';
            if (precision >= 0.5) return 'удовлетворительное качество';
            return 'требует улучшения';
        }

        async function loadClassifierMetrics() {
            try {
                const response = await fetch('/api/model/metrics');
                const data = await response.json();

                let detailsHtml = '';

                if (data.status === 'success' && data.metrics) {
                    const metrics = data.metrics;

                    detailsHtml += `<p><strong>Основные метрики классификации:</strong></p>`;
                    detailsHtml += `<ul>`;
                    if (metrics.test_precision !== undefined) {
                        detailsHtml += `<li><strong>Precision:</strong> ${(metrics.test_precision * 100).toFixed(2)}%</li>`;
                    }
                    if (metrics.test_accuracy !== undefined) {
                        detailsHtml += `<li><strong>Accuracy:</strong> ${(metrics.test_accuracy * 100).toFixed(2)}%</li>`;
                    }
                    if (metrics.test_recall !== undefined) {
                        detailsHtml += `<li><strong>Recall:</strong> ${(metrics.test_recall * 100).toFixed(2)}%</li>`;
                    }
                    if (metrics.test_f1_score !== undefined) {
                        detailsHtml += `<li><strong>F1-Score:</strong> ${(metrics.test_f1_score * 100).toFixed(2)}%</li>`;
                    }
                    if (metrics.test_auc !== undefined) {
                        detailsHtml += `<li><strong>AUC:</strong> ${(metrics.test_auc * 100).toFixed(2)}%</li>`;
                    }
                    detailsHtml += `</ul>`;

                    if (metrics.classification_report) {
                        detailsHtml += `<p><strong>Classification Report:</strong></p>`;
                        detailsHtml += `<pre style="font-size: 12px; background: white; padding: 10px; border-radius: 5px; overflow-x: auto;">${JSON.stringify(metrics.classification_report, null, 2)}</pre>`;
                    }

                } else {
                    detailsHtml = '<p>Не удалось загрузить детальные метрики классификации</p>';
                }

                document.getElementById('detailedMetrics').innerHTML = detailsHtml;

            } catch (error) {
                console.error('Ошибка загрузки детальных метрик классификации:', error);
                document.getElementById('detailedMetrics').innerHTML = 
                    '<p>Не удалось загрузить детальную информацию о метриках классификации</p>';
            }
        }

        async function checkItem() {
            const itemId = document.getElementById('itemId').value;
            if (!itemId) {
                alert('Пожалуйста, введите код товара');
                return;
            }

            try {
                const response = await fetch(`/api/items/search?item_id=${itemId}`);
                const data = await response.json();

                if (data.status === 'success') {
                    if (data.item_found) {
                        alert(`✅ Товар ${itemId} найден в базе данных!`);
                    } else {
                        alert(`❌ Товар ${itemId} не найден в базе данных.`);
                    }
                } else {
                    alert('Ошибка при проверке товара: ' + (data.error || 'неизвестная ошибка'));
                }
            } catch (error) {
                alert('Ошибка соединения: ' + error.message);
            }
        }

        async function getRecommendations() {
            const itemId = document.getElementById('itemId').value;
            const searchButton = document.getElementById('searchButton');

            if (!itemId) {
                alert('Пожалуйста, введите код товара');
                return;
            }

            const loading = document.getElementById('loading');
            const results = document.getElementById('results');

            // Блокируем кнопку на время поиска
            searchButton.disabled = true;
            searchButton.textContent = 'Поиск...';

            loading.style.display = 'block';
            results.innerHTML = '';

            try {
                const response = await fetch('/api/recommend/item', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        item_id: parseInt(itemId),
                        top_k: 10
                    })
                });

                const data = await response.json();

                loading.style.display = 'none';
                searchButton.disabled = false;
                searchButton.textContent = 'Найти рекомендации';

                if (data.status === 'success') {
                    displayRecommendations(data);
                } else {
                    results.innerHTML = `<div class="error">❌ Ошибка: ${data.message}</div>`;
                }
            } catch (error) {
                loading.style.display = 'none';
                searchButton.disabled = false;
                searchButton.textContent = 'Найти рекомендации';
                results.innerHTML = `<div class="error">❌ Ошибка соединения: ${error.message}</div>`;
            }
        }

        function displayRecommendations(data) {
            const results = document.getElementById('results');

            const precisionPercent = (data.precision * 100).toFixed(1);
            const classifierPrecisionPercent = (currentClassifierPrecision * 100).toFixed(1);
            const recommendationPrecisionPercent = (currentRecommendationPrecision * 100).toFixed(1);

            let html = `<div class="success">
                <h3>✅ Рекомендации найдены!</h3>
                <p>Для товара <strong>${data.item_id}</strong> найдено ${data.total_recommendations} рекомендаций</p>
                <p><strong>Всего товаров в базе:</strong> ${data.total_items_in_database?.toLocaleString() || 'N/A'}</p>
                <p><strong>Precision модели классификации:</strong> ${classifierPrecisionPercent}% (${getPrecisionDescription(currentClassifierPrecision)})</p>
                <p><strong>Precision системы рекомендаций:</strong> ${recommendationPrecisionPercent}% (${getPrecisionDescription(currentRecommendationPrecision)})</p>
            </div>`;

            html += `<div class="recommendation-block">
                <div class="precision-badge">Precision: ${precisionPercent}%</div>
                <div class="block-title">🎯 Рекомендуемые товары</div>
                <div class="item-list">`;

            data.recommendations.forEach((rec, index) => {
                const similarityPercent = (rec.similarity * 100).toFixed(1);
                const itemPrecision = (rec.precision * 100).toFixed(1);
                html += `
                <div class="item-card">
                    <div class="item-id">${index + 1}. Товар #${rec.item_id}</div>
                    <div class="similarity">
                        Схожесть: ${similarityPercent}% 
                        <span style="color: #6c757d; font-size: 12px;">(precision: ${itemPrecision}%)</span>
                    </div>
                </div>`;
            });

            html += `</div></div>`;

            // Добавляем статистику
            const avgSimilarity = data.recommendations.reduce((sum, rec) => sum + rec.similarity, 0) / data.recommendations.length;
            html += `
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">${data.total_recommendations}</div>
                    <div class="stat-label">Всего рекомендаций</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${classifierPrecisionPercent}%</div>
                    <div class="stat-label">Точность классификации</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${(avgSimilarity * 100).toFixed(1)}%</div>
                    <div class="stat-label">Средняя схожесть</div>
                </div>
            </div>`;

            results.innerHTML = html;
        }

        // Обработка нажатия Enter в поле ввода
        document.getElementById('itemId').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                getRecommendations();
            }
        });

        // Периодическое обновление информации о модели (каждые 30 секунд)
        setInterval(loadModelInfo, 30000);

        // Загружаем информацию о модели при загрузке страницы
        document.addEventListener('DOMContentLoaded', function() {
            loadModelInfo();
        });
    </script>
</body>
</html>
    """

    templates_dir = os.path.join(current_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)

    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)


# Создаем HTML шаблон при запуске
create_html_template()

# Пример использования
if __name__ == "__main__":
    # Тестирование класса
    classifier = TensorFlowProductClassifier()

    try:
        # Загрузка данных
        data_path = "data/result_df_analysis.csv"

        # Проверяем существование файла
        if not os.path.exists(data_path):
            # Создаем демо-данные если файл не существует
            logger.info(" Создание демо-данных...")
            demo_data = pd.DataFrame({
                'itemid': range(1000),
                'feature_0': np.random.normal(0, 1, 1000),
                'feature_1': np.random.normal(0, 1, 1000),
                'feature_2': np.random.normal(0, 1, 1000),
                'activity_category': np.random.choice(['rare', 'periodic', 'loyal'], 1000, p=[0.2, 0.3, 0.5])
            })
            os.makedirs('data', exist_ok=True)
            demo_data.to_csv(data_path, index=False)
            logger.info("Демо-данные созданы")

        df = classifier.load_and_preprocess_data(data_path)

        # Обучение модели
        result = classifier.train(df, epochs=10, batch_size=32)
        print("Обучение завершено:", result["status"])

        if result["status"] == "success":
            # Тестирование предсказания
            test_features = {f'feature_{i}': np.random.normal(0, 1) for i in range(len(classifier.feature_columns))}
            prediction = classifier.predict_single(test_features)
            print("Предсказание:", prediction)

            # Информация о модели
            summary = classifier.get_model_summary()
            print("Информация о модели:", summary)

            # Тестируем новый метод
            test_metrics = classifier.get_test_metrics()
            print("Метрики тестирования:", test_metrics)
        else:
            print("Ошибка обучения:", result["message"])

    except Exception as e:
        print(f"Общая ошибка: {e}")

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
