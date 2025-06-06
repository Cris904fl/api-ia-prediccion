#!/usr/bin/env python3
"""
Modelo de predicción de ventas usando Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class SalesPredictor:
    """Clase para predicción de ventas"""
    
    def __init__(self):
        """Inicializar el predictor de ventas"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'producto_encoded', 'temporada_encoded', 
            'precio', 'stock', 'precio_normalizado'
        ]
        
        # Intentar cargar modelo pre-entrenado
        self._load_model()
    
    def _encode_categorical_features(self, data, fit=False):
        """Codificar características categóricas"""
        encoded_data = data.copy()
        categorical_columns = ['producto', 'temporada']
        
        for column in categorical_columns:
            if column in encoded_data.columns:
                if fit or column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    encoded_data[f'{column}_encoded'] = self.label_encoders[column].fit_transform(
                        encoded_data[column].astype(str)
                    )
                else:
                    # Manejar valores no vistos durante el entrenamiento
                    try:
                        encoded_data[f'{column}_encoded'] = self.label_encoders[column].transform(
                            encoded_data[column].astype(str)
                        )
                    except ValueError:
                        # Asignar valor por defecto para categorías no vistas
                        logger.warning(f"Valor no visto en {column}: {encoded_data[column].iloc[0]}")
                        encoded_data[f'{column}_encoded'] = 0
        
        return encoded_data
    
    def _create_features(self, data):
        """Crear características adicionales"""
        featured_data = data.copy()
        
        # Normalizar precio
        if 'precio' in featured_data.columns:
            featured_data['precio_normalizado'] = featured_data['precio'] / 1000
        
        # Crear ratio stock/precio si es relevante
        if 'stock' in featured_data.columns and 'precio' in featured_data.columns:
            featured_data['stock_precio_ratio'] = featured_data['stock'] / (featured_data['precio'] + 1)
        
        return featured_data
    
    def train(self, training_data):
        """Entrenar el modelo con datos de entrenamiento"""
        try:
            if not training_data:
                # Generar datos de ejemplo para demostración
                training_data = self._generate_sample_data()
            
            # Convertir a DataFrame
            df = pd.DataFrame(training_data)
            
            # Validar columnas requeridas
            required_columns = ['producto', 'temporada', 'precio', 'stock', 'ventas']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Columnas faltantes: {missing_columns}")
            
            # Preprocesar datos
            df = self._encode_categorical_features(df, fit=True)
            df = self._create_features(df)
            
            # Preparar características y objetivo
            X = df[self.feature_columns]
            y = df['ventas']
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Escalar características
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar modelo
            self.model.fit(X_train_scaled, y_train)
            
            # Validar modelo
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            # Guardar modelo
            self._save_model()
            
            return {
                'success': True,
                'message': 'Modelo entrenado exitosamente',
                'metrics': {
                    'mse': float(mse),
                    'r2_score': float(r2),
                    'samples_trained': len(training_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, data):
        """Realizar predicción de ventas"""
        try:
            if not self.is_trained:
                # Si no hay modelo entrenado, usar uno básico
                return self._basic_prediction(data)
            
            # Convertir a DataFrame
            df = pd.DataFrame([data])
            
            # Preprocesar datos
            df = self._encode_categorical_features(df)
            df = self._create_features(df)
            
            # Preparar características
            X = df[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            
            # Realizar predicción
            prediction = self.model.predict(X_scaled)[0]
            
            # Calcular confianza basada en características del producto
            confidence = self._calculate_confidence(data, prediction)
            
            return {
                'ventas_predichas': max(0, int(prediction)),
                'confianza': confidence,
                'factores': self._analyze_factors(data),
                'recomendaciones': self._generate_recommendations(data, prediction)
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return self._basic_prediction(data)
    
    def _basic_prediction(self, data):
        """Predicción básica sin modelo entrenado"""
        # Lógica simple basada en reglas de negocio
        base_sales = 100
        
        # Ajustar por precio
        precio = data.get('precio', 1000)
        if precio < 500:
            price_factor = 1.5
        elif precio < 2000:
            price_factor = 1.2
        else:
            price_factor = 0.8
        
        # Ajustar por temporada
        temporada = data.get('temporada', '').lower()
        season_factors = {
            'alta': 1.5,
            'media': 1.0,
            'baja': 0.7,
            'navidad': 2.0,
            'verano': 1.3
        }
        season_factor = season_factors.get(temporada, 1.0)
        
        # Ajustar por stock
        stock = data.get('stock', 50)
        stock_factor = min(1.0, stock / 100)
        
        predicted_sales = int(base_sales * price_factor * season_factor * stock_factor)
        
        return {
            'ventas_predichas': max(1, predicted_sales),
            'confianza': 0.6,
            'factores': {
                'precio': price_factor,
                'temporada': season_factor,
                'stock': stock_factor
            },
            'recomendaciones': [
                'Modelo básico - Entrenar con datos reales para mejores predicciones',
                f'Stock actual: {stock} unidades',
                f'Temporada: {temporada}'
            ]
        }
    
    def _calculate_confidence(self, data, prediction):
        """Calcular confianza de la predicción"""
        confidence = 0.8
        
        # Reducir confianza si hay valores extremos
        precio = data.get('precio', 0)
        if precio > 10000 or precio < 100:
            confidence -= 0.2
        
        stock = data.get('stock', 0)
        if stock > 1000 or stock < 1:
            confidence -= 0.1
        
        return max(0.3, confidence)
    
    def _analyze_factors(self, data):
        """Analizar factores que influyen en la predicción"""
        return {
            'precio': data.get('precio', 0),
            'stock_disponible': data.get('stock', 0),
            'temporada': data.get('temporada', 'No especificada'),
            'producto': data.get('producto', 'No especificado')
        }
    
    def _generate_recommendations(self, data, prediction):
        """Generar recomendaciones basadas en la predicción"""
        recommendations = []
        
        precio = data.get('precio', 0)
        stock = data.get('stock', 0)
        
        if prediction > stock:
            recommendations.append(f"Considerar aumentar el stock (actual: {stock}, predicción: {int(prediction)})")
        
        if precio > 5000:
            recommendations.append("Precio alto - considerar promociones para aumentar ventas")
        elif precio < 500:
            recommendations.append("Precio bajo - verificar márgenes de ganancia")
        
        recommendations.append(f"Mantener seguimiento de ventas para temporada: {data.get('temporada', 'actual')}")
        
        return recommendations
    
    def _generate_sample_data(self):
        """Generar datos de ejemplo para entrenamiento"""
        np.random.seed(42)
        
        productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Tablet', 'Smartphone']
        temporadas = ['alta', 'media', 'baja', 'navidad']
        
        sample_data = []
        
        for _ in range(200):
            producto = np.random.choice(productos)
            temporada = np.random.choice(temporadas)
            precio = np.random.randint(200, 8000)
            stock = np.random.randint(10, 500)
            
            # Generar ventas basadas en lógica simple
            base_ventas = 50
            if temporada == 'navidad':
                base_ventas *= 2
            elif temporada == 'alta':
                base_ventas *= 1.5
            elif temporada == 'baja':
                base_ventas *= 0.7
            
            if precio < 1000:
                base_ventas *= 1.3
            elif precio > 5000:
                base_ventas *= 0.8
            
            ventas = int(base_ventas + np.random.normal(0, 10))
            ventas = max(1, min(ventas, stock))
            
            sample_data.append({
                'producto': producto,
                'temporada': temporada,
                'precio': precio,
                'stock': stock,
                'ventas': ventas
            })
        
        return sample_data
    
    def _save_model(self):
        """Guardar modelo entrenado"""
        try:
            os.makedirs('models/saved', exist_ok=True)
            
            joblib.dump(self.model, 'models/saved/sales_model.pkl')
            joblib.dump(self.label_encoders, 'models/saved/label_encoders.pkl')
            joblib.dump(self.scaler, 'models/saved/scaler.pkl')
            
            logger.info("Modelo guardado exitosamente")
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
    
    def _load_model(self):
        """Cargar modelo pre-entrenado"""
        try:
            if os.path.exists('models/saved/sales_model.pkl'):
                self.model = joblib.load('models/saved/sales_model.pkl')
                self.label_encoders = joblib.load('models/saved/label_encoders.pkl')
                self.scaler = joblib.load('models/saved/scaler.pkl')
                self.is_trained = True
                logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.info(f"No se pudo cargar modelo pre-entrenado: {str(e)}")