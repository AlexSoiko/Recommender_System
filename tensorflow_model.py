import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
import joblib
import os
import logging

logger = logging.getLogger(__name__)


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

    def load_and_preprocess_data(self, data_path: str):
        """Загрузка и предобработка данных"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Данные загружены: {df.shape}")

            # Создаем фиктивную целевую переменную если ее нет
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

    def build_simple_model(self, input_dim, num_classes=3):
        """Простая и стабильная архитектура модели"""
        model = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Простая модель построена")
        return model

    def train(self, df, epochs=20, batch_size=32, validation_split=0.2, test_size=0.2):
        """Обучение модели"""
        try:
            X = df[self.feature_columns].values.astype('float32')
            y = df['activity_category_encoded'].values

            logger.info(f"Размерность данных: X={X.shape}, y={y.shape}")

            X_scaled = self.scaler.fit_transform(X)

            # Разделение данных
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_scaled, y, test_size=test_size, stratify=y, random_state=42
            )

            val_size_adjusted = validation_split / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
            )

            # Построение и обучение модели
            self.model = self.build_simple_model(X_train.shape[1], len(self.label_encoder.classes_))

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                )
            ]

            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Оценка модели
            self.test_metrics = self.evaluate_with_sklearn(X_test, y_test)
            self.is_trained = True
            self.save_model()

            return {
                "status": "success",
                "training_history": self.history.history,
                "test_metrics": self.test_metrics
            }

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            return {"status": "error", "message": str(e)}

    def evaluate_with_sklearn(self, X_test, y_test):
        """Оценка модели с использованием sklearn"""
        try:
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

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

            logger.info("Метрики на тестовой выборке:")
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
            features_list = []
            for col in self.feature_columns:
                if col in item_features:
                    features_list.append(item_features[col])
                else:
                    features_list.append(0.0)

            features_array = np.array([features_list]).astype('float32')
            features_scaled = self.scaler.transform(features_array)

            probabilities = self.model.predict(features_scaled, verbose=0)
            predicted_class = np.argmax(probabilities, axis=1)[0]
            confidence = np.max(probabilities, axis=1)[0]

            predicted_category = self.label_encoder.inverse_transform([predicted_class])[0]

            class_probabilities = {
                self.label_encoder.inverse_transform([i])[0]: float(probabilities[0][i])
                for i in range(len(self.label_encoder.classes_))
            }

            return {
                'category': predicted_category,
                'confidence': float(confidence),
                'class_probabilities': class_probabilities,
                'all_categories': self.label_encoder.classes_.tolist()
            }

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return {"error": f"Prediction error: {str(e)}"}

    def save_model(self):
        """Сохранение модели"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)

            preprocessing_path = self.model_path.replace('.h5', '_preprocessing.pkl')
            joblib.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'test_metrics': self.test_metrics
            }, preprocessing_path)

            logger.info("Модель сохранена")

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
            else:
                logger.warning("Файл препроцессинга не найден")

            logger.info("Модель загружена успешно")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def get_test_metrics(self):
        """Получение метрик тестирования модели"""
        return self.test_metrics

    def get_model_summary(self):
        """Получение информации о модели"""
        if not self.model:
            return {"error": "Model not loaded"}

        return {
            "is_trained": self.is_trained,
            "feature_columns": self.feature_columns,
            "num_classes": len(self.label_encoder.classes_),
            "classes": self.label_encoder.classes_.tolist(),
            "test_metrics": self.test_metrics
        }

