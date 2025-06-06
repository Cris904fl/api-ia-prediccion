#!/usr/bin/env python3
"""
API Flask para predicción de ventas y clasificación de productos
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

# Importar modelos
from models.sales_predictor import SalesPredictor
from models.product_classifier import ProductClassifier

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)

# Configuración
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Inicializar modelos
sales_predictor = SalesPredictor()
product_classifier = ProductClassifier()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del sistema"""
    return jsonify({
        'status': 'OK',
        'message': 'API funcionando correctamente',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/predict-sales', methods=['POST'])
def predict_sales():
    """Endpoint para predicción de ventas"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No se proporcionaron datos',
                'status': 'error'
            }), 400
        
        # Validar datos requeridos
        required_fields = ['producto', 'temporada', 'precio', 'stock']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Campos faltantes: {", ".join(missing_fields)}',
                'status': 'error'
            }), 400
        
        # Realizar predicción
        prediction = sales_predictor.predict(data)
        
        return jsonify({
            'prediction': prediction,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en predicción de ventas: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/classify-products', methods=['POST'])
def classify_products():
    """Endpoint para clasificación de productos"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No se proporcionaron datos',
                'status': 'error'
            }), 400
        
        # Validar datos requeridos
        if 'descripcion' not in data:
            return jsonify({
                'error': 'Campo "descripcion" es requerido',
                'status': 'error'
            }), 400
        
        # Realizar clasificación
        classification = product_classifier.classify(data['descripcion'])
        
        return jsonify({
            'classification': classification,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en clasificación de productos: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Endpoint para entrenar modelos"""
    try:
        data = request.get_json()
        
        if not data or 'model_type' not in data:
            return jsonify({
                'error': 'Tipo de modelo no especificado',
                'status': 'error'
            }), 400
        
        model_type = data['model_type']
        training_data = data.get('training_data', [])
        
        if model_type == 'sales':
            result = sales_predictor.train(training_data)
        elif model_type == 'classification':
            result = product_classifier.train(training_data)
        else:
            return jsonify({
                'error': 'Tipo de modelo no válido',
                'status': 'error'
            }), 400
        
        return jsonify({
            'result': result,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'status': 'error',
            'details': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Manejador de errores 404"""
    return jsonify({
        'error': 'Endpoint no encontrado',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Manejador de errores 500"""
    return jsonify({
        'error': 'Error interno del servidor',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)