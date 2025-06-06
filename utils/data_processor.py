import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.default_values = {
            'price': 0,
            'promotion': 0,
            'season': 1,
            'category': 1,
            'stock_level': 100,
            'competitor_price': 0,
            'marketing_spend': 0,
            'avg_monthly_sales': 0,
            'stock_turnover': 1,
            'seasonality_factor': 1,
            'category_popularity': 1,
            'reviews_count': 0,
            'avg_rating': 0,
            'discount_frequency': 0,
            'competitor_count': 1,
            'marketing_investment': 0
        }
    
    def prepare_sales_data(self, data):
        """Preparar datos para predicción de ventas"""
        processed_data = {}
        
        # Campos básicos requeridos
        required_fields = ['product_id', 'month', 'year']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Campo requerido faltante: {field}")
            processed_data[field] = data[field]
        
        # Campos opcionales con valores por defecto
        for field, default_value in self.default_values.items():
            processed_data[field] = data.get(field, default_value)
        
        # Validaciones
        processed_data['month'] = max(1, min(12, processed_data['month']))
        processed_data['year'] = max(2020, min(2030, processed_data['year']))
        processed_data['price'] = max(0, processed_data['price'])
        processed_data['stock_level'] = max(0, processed_data['stock_level'])
        
        # Determinar estación automáticamente si no se proporciona
        if 'season' not in data:
            month = processed_data['month']
            if month in [12, 1, 2]:
                processed_data['season'] = 1  # Invierno
            elif month in [3, 4, 5]:
                processed_data['season'] = 2  # Primavera
            elif month in [6, 7, 8]:
                processed_data['season'] = 3  # Verano
            else:
                processed_data['season'] = 4  # Otoño
        
        return processed_data
    
    def prepare_classification_data(self, data):
        """Preparar datos para clasificación de productos"""
        processed_data = {}
        
        # Campos básicos
        processed_data['product_id'] = data.get('product_id', 'unknown')
        
        # Campos para clasificación con valores por defecto
        classification_fields = [
            'avg_monthly_sales', 'price', 'stock_turnover', 'seasonality_factor',
            'category_popularity', 'reviews_count', 'avg_rating', 'discount_frequency',
            'competitor_count', 'marketing_investment'
        ]
        
        for field in classification_fields:
            processed_data[field] = data.get(field, self.default_values.get(field, 0))
        
        # Campos adicionales calculados
        processed_data['avg_category_price'] = data.get('avg_category_price', processed_data['price'])
        processed_data['stock_level'] = data.get('stock_level', 100)
        processed_data['is_seasonal'] = data.get('is_seasonal', False)
        
        # Validaciones
        processed_data['price'] = max(0, processed_data['price'])
        processed_data['avg_monthly_sales'] = max(0, processed_data['avg_monthly_sales'])
        processed_data['stock_turnover'] = max(0, processed_data['stock_turnover'])
        processed_data['avg_rating'] = max(0, min(5, processed_data['avg_rating']))
        processed_data['reviews_count'] = max(0, processed_data['reviews_count'])
        
        return processed_data
    
    def prepare_training_sales_data(self, training_data):
        """Preparar datos de entrenamiento para predicción de ventas"""
        processed_data = []
        
        for item in training_data:
            try:
                # Validar que tenga los campos necesarios para entrenamiento
                if 'sales' not in item:
                    continue
                
                processed_item = self.prepare_sales_data(item)
                processed_item['sales'] = max(0, item['sales'])  # Target variable
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"Error procesando item de entrenamiento: {e}")
                continue
        
        if not processed_data:
            raise ValueError("No se pudieron procesar datos de entrenamiento válidos")
        
        return pd.DataFrame(processed_data)
    
    def prepare_training_classification_data(self, training_data):
        """Preparar datos de entrenamiento para clasificación"""
        processed_data = []
        
        for item in training_data:
            try:
                processed_item = self.prepare_classification_data(item)
                
                # Asegurar que tenga ventas promedio para determinar la clase
                if 'avg_monthly_sales' not in processed_item or processed_item['avg_monthly_sales'] == 0:
                    # Si no hay ventas específicas, usar sales si está disponible
                    if 'sales' in item:
                        processed_item['avg_monthly_sales'] = max(0, item['sales'])
                
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"Error procesando item de clasificación: {e}")
                continue
        
        if not processed_data:
            raise ValueError("No se pudieron procesar datos de clasificación válidos")
        
        return pd.DataFrame(processed_data)
    
    def validate_data_format(self, data, data_type='sales'):
        """Validar formato de datos de entrada"""
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Los datos deben ser un diccionario")
            return errors
        
        if data_type == 'sales':
            required_fields = ['product_id', 'month', 'year']
            for field in required_fields:
                if field not in data:
                    errors.append(f"Campo requerido faltante: {field}")
                elif not isinstance(data[field], (int, float)):
                    errors.append(f"Campo {field} debe ser numérico")
        
        elif data_type == 'classification':
            if 'product_id' not in data:
                errors.append("Campo product_id es requerido")
        
        # Validaciones de rangos
        if 'month' in data:
            if not (1 <= data['month'] <= 12):
                errors.append("El mes debe estar entre 1 y 12")
        
        if 'year' in data:
            if not (2020 <= data['year'] <= 2030):
                errors.append("El año debe estar entre 2020 y 2030")
        
        if 'price' in data:
            if data['price'] < 0:
                errors.append("El precio no puede ser negativo")
        
        return errors
    
    def clean_numeric_data(self, value, default=0):
        """Limpiar y convertir datos numéricos"""
        try:
            if value is None or value == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def normalize_category(self, category):
        """Normalizar categorías de productos"""
        category_mapping = {
            'electronics': 1,
            'electrónicos': 1,
            'clothing': 2,
            'ropa': 2,
            'home': 3,
            'hogar': 3,
            'books': 4,
            'libros': 4,
            'sports': 5,
            'deportes': 5,
            'health': 6,
            'salud': 6,
            'beauty': 7,
            'belleza': 7,
            'food': 8,
            'comida': 8,
            'toys': 9,
            'juguetes': 9,
            'automotive': 10,
            'automotriz': 10
        }
        
        if isinstance(category, str):
            return category_mapping.get(category.lower(), 1)
        elif isinstance(category, (int, float)):
            return max(1, min(10, int(category)))
        else:
            return 1
    
    def calculate_derived_features(self, data):
        """Calcular características derivadas"""
        derived = {}
        
        # Ratio precio vs competencia
        if data.get('price', 0) > 0 and data.get('competitor_price', 0) > 0:
            derived['price_ratio'] = data['price'] / data['competitor_price']
        else:
            derived['price_ratio'] = 1.0
        
        # Índice de popularidad
        reviews = data.get('reviews_count', 0)
        rating = data.get('avg_rating', 0)
        derived['popularity_index'] = (reviews * rating) / 1000 if reviews > 0 and rating > 0 else 0
        
        # Factor de estacionalidad
        month = data.get('month', 1)
        if month in [11, 12, 1]:  # Temporada navideña
            derived['seasonal_boost'] = 1.3
        elif month in [6, 7, 8]:  # Temporada de verano
            derived['seasonal_boost'] = 1.1
        else:
            derived['seasonal_boost'] = 1.0
        
        # Índice de rotación de inventario
        sales = data.get('avg_monthly_sales', 0)
        stock = data.get('stock_level', 1)
        derived['inventory_turnover'] = sales / stock if stock > 0 else 0
        
        return derived