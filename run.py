#!/usr/bin/env python3
"""
Script de ejecución para la API de predicción de ventas y clasificación de productos
"""

import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importar la aplicación Flask
from app import app

def main():
    """Función principal para ejecutar la aplicación"""
    
    # Configuración del servidor
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("🚀 INICIANDO API DE PREDICCIÓN DE VENTAS")
    print("=" * 60)
    print(f"📍 Host: {host}")
    print(f"🔌 Puerto: {port}")
    print(f"🐛 Debug: {debug}")
    print("\n📋 Endpoints disponibles:")
    print("   GET  /api/health              - Estado del sistema")
    print("   POST /api/predict-sales       - Predicción de ventas")
    print("   POST /api/classify-products   - Clasificación de productos")
    print("   POST /api/train-model         - Entrenar modelos")
    print("\n💡 Ejemplos de uso:")
    print("   curl http://localhost:5000/api/health")
    print("=" * 60)
    
    try:
        # Ejecutar la aplicación
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error al iniciar el servidor: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()