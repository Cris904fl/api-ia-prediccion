import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProductClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.is_model_trained = False
        self.confidence_score = 0.0
        self.classes = ['Baja Demanda', 'Demanda Moderada', 'Alta Demanda']
        self.feature_names = [
            'avg_monthly_sales', 'price', 'stock_turnover', 'seasonality_factor',
            'category_popularity', 'reviews_count', 'avg_rating', 'discount_frequency',
            'competitor_count', 'marketing_investment', 'price_ratio', 'rotation_ratio',
            'is_seasonal'
        ]
        self.model_path = 'models/saved/product_classifier.joblib'
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Intentar cargar modelo pre-entrenado
        self.load_model()
    
    def prepare_features(self, data):
        """Preparar características para clasificación"""
        features = []
        
        # Características del producto
        features.extend([
            data.get('avg_monthly_sales', 0),
            data.get('price', 0),
            data.get('stock_turnover', 0),
            data.get('seasonality_factor', 1),
            data.get('category_popularity', 1),
            data.get('reviews_count', 0),
            data.get('avg_rating', 0),
            data.get('discount_frequency', 0),
            data.get('competitor_count', 1),
            data.get('marketing_investment', 0)
        ])
        
        # Características derivadas
        price = data.get('price', 1)
        avg_category_price = data.get('avg_category_price', price)
        features.append(price / max(avg_category_price, 1))  # Ratio precio
        
        sales = data.get('avg_monthly_sales', 1)
        stock_level = data.get('stock_level', 1)
        features.append(sales / max(stock_level, 1))  # Rotación
        
        features.append(1 if data.get('is_seasonal', False) else 0)  # Estacionalidad
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data):
        """Entrenar el modelo clasificador"""
        try:
            if isinstance(training_data, list):
                df = pd.DataFrame(training_data)
            else:
                df = training_data.copy()
            
            # Preparar características y etiquetas
            X = []
            y = []
            
            for _, row in df.iterrows():
                features = self.prepare_features(row.to_dict())[0]
                X.append(features)
                
                # Determinar clase basada en ventas promedio
                sales = row.get('avg_monthly_sales', 0)
                if sales < 50:
                    label = 0  # Baja Demanda
                elif sales < 150:
                    label = 1  # Demanda Moderada
                else:
                    label = 2  # Alta Demanda
                
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Validar datos
            if len(X) < 10:
                raise ValueError("Se necesitan al menos 10 muestras para entrenar")
            
            # Asegurar que todas las clases estén representadas
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                raise ValueError("Se necesitan al menos 2 clases diferentes")
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Entrenar modelo
            self.model.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.confidence_score = accuracy
            self.is_model_trained = True
            
            # Guardar modelo
            self.save_model()
            
            return {
                'status': 'success',
                'accuracy': float(accuracy),
                'samples_trained': len(X_train),
                'classes': self.classes,
                'confidence': float(self.confidence_score),
                'feature_importance': {
                    name: float(importance) 
                    for name, importance in zip(self.feature_names, self.model.feature_importances_)
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def classify(self, data):
        """Clasificar producto por demanda"""
        if not self.is_model_trained:
            # Entrenar con datos de ejemplo si no hay modelo
            self._train_with_sample_data()
        
        try:
            features = self.prepare_features(data)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Calcular confianza
            max_prob = np.max(probabilities)
            confidence_level = 'Alta' if max_prob > 0.7 else 'Media' if max_prob > 0.5 else 'Baja'
            
            result = {
                'class': self.classes[prediction],
                'class_id': int(prediction),
                'confidence': float(max_prob),
                'confidence_level': confidence_level,
                'probabilities': {
                    self.classes[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                },
                'recommendations': self._get_recommendations(prediction, data)
            }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_recommendations(self, prediction, data):
        """Generar recomendaciones basadas en la clasificación"""
        recommendations = []
        
        if prediction == 0:  # Baja Demanda
            recommendations.extend([
                "Considerar reducir precio para aumentar demanda",
                "Implementar estrategias de marketing más agresivas",
                "Evaluar mejoras en el producto basadas en reviews",
                "Considerar descuentos temporales para mover inventario"
            ])
        elif prediction == 1:  # Demanda Moderada
            recommendations.extend([
                "Mantener niveles actuales de stock",
                "Considerar optimización de precios",
                "Monitorear competencia cercana",
                "Evaluar oportunidades de cross-selling"
            ])
        else:  # Alta Demanda
            recommendations.extend([
                "Asegurar disponibilidad de stock suficiente",
                "Considerar incremento de precio gradual",
                "Expandir marketing para mantener momentum",
                "Evaluar oportunidades de productos complementarios"
            ])
        
        # Recomendaciones específicas basadas en datos
        if data.get('avg_rating', 0) < 3.5:
            recommendations.append("Mejorar calidad del producto (rating bajo)")
        
        if data.get('reviews_count', 0) < 10:
            recommendations.append("Incentivar más reviews de clientes")
        
        if data.get('stock_turnover', 0) < 0.5:
            recommendations.append("Mejorar rotación de inventario")
        
        return recommendations
    
    def get_confidence_score(self):
        """Obtener score de confianza del modelo"""
        return self.confidence_score
    
    def is_trained(self):
        """Verificar si el modelo está entrenado"""
        return self.is_model_trained
    
    def save_model(self):
        """Guardar modelo entrenado"""
        try:
            model_data = {
                'model': self.model,
                'confidence_score': self.confidence_score,
                'is_trained': self.is_model_trained,
                'classes': self.classes,
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, self.model_path)
            print(f"Modelo clasificador guardado en: {self.model_path}")
        except Exception as e:
            print(f"Error guardando modelo clasificador: {e}")
    
    def load_model(self):
        """Cargar modelo pre-entrenado"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.confidence_score = model_data.get('confidence_score', 0.0)
                self.is_model_trained = model_data.get('is_trained', False)
                self.classes = model_data.get('classes', self.classes)
                self.feature_names = model_data.get('feature_names', self.feature_names)
                print("Modelo clasificador cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo clasificador: {e}")
            self.is_model_trained = False
    
    def _train_with_sample_data(self):
        """Entrenar con datos de ejemplo para demostración"""
        print("Entrenando clasificador con datos de ejemplo...")
        
        # Datos de ejemplo con patrones realistas
        sample_data = []
        
        for i in range(300):
            # Crear diferentes patrones de demanda
            if i < 100:  # Baja demanda
                base_sales = np.random.normal(25, 10)
                price_factor = 1.2  # Precios más altos
                rating_factor = 0.8  # Ratings más bajos
            elif i < 200:  # Demanda moderada
                base_sales = np.random.normal(100, 15)
                price_factor = 1.0
                rating_factor = 1.0
            else:  # Alta demanda
                base_sales = np.random.normal(200, 25)
                price_factor = 0.9  # Precios más competitivos
                rating_factor = 1.2  # Ratings más altos
            
            sample_data.append({
                'product_id': i,
                'avg_monthly_sales': max(0, base_sales + np.random.normal(0, 5)),
                'price': max(1, 50 * price_factor + np.random.normal(0, 10)),
                'stock_turnover': max(0.1, np.random.normal(1.0, 0.3)),
                'seasonality_factor': 0.8 + np.random.random() * 0.4,
                'category_popularity': 1 + np.random.randint(0, 10),
                'reviews_count': max(0, int(np.random.normal(100, 50))),
                'avg_rating': max(1, min(5, 3 + np.random.normal(0, 1) * rating_factor)),
                'discount_frequency': np.random.random() * 0.3,
                'competitor_count': 1 + np.random.randint(0, 20),
                'marketing_investment': max(0, np.random.normal(1000, 500)),
                'avg_category_price': 55 + np.random.normal(0, 15),
                'stock_level': 50 + np.random.randint(0, 200),
                'is_seasonal': np.random.choice([True, False], p=[0.3, 0.7])
            })
        
        result = self.train(sample_data)
        if result['status'] == 'success':
            print(f"Entrenamiento completado. Precisión: {result['accuracy']:.3f}")
        else:
            print(f"Error en entrenamiento: {result['message']}")
    
    def get_feature_importance(self):
        """Obtener importancia de características"""
        if not self.is_model_trained:
            return {}
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(importance)
        
        # Ordenar por importancia
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def validate_input(self, data):
        """Validar datos de entrada"""
        required_fields = ['price']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                'valid': False,
                'message': f"Campos requeridos faltantes: {missing_fields}"
            }
        
        # Validar rangos
        if data.get('price', 0) <= 0:
            return {
                'valid': False,
                'message': "El precio debe ser mayor que 0"
            }
        
        return {'valid': True}
    
    def export_model_info(self):
        """Exportar información del modelo"""
        if not self.is_model_trained:
            return {"error": "Modelo no entrenado"}
        
        return {
            'model_type': 'RandomForestClassifier',
            'classes': self.classes,
            'feature_names': self.feature_names,
            'confidence_score': self.confidence_score,
            'is_trained': self.is_model_trained,
            'feature_importance': self.get_feature_importance(),
            'model_parameters': self.model.get_params()
        }