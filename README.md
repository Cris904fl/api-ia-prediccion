# 🤖 API de Predicción de Ventas con IA

API REST desarrollada en Python con Flask que proporciona funcionalidades de Machine Learning para predicción de ventas y clasificación de productos por demanda.

## 📋 Características Principales

### 🔮 Predicción de Ventas
- Predicción de ventas futuras usando Regresión Lineal
- Análisis de tendencias y patrones estacionales
- Recomendaciones automáticas basadas en datos históricos
- Nivel de confianza de predicciones

### 📊 Clasificación de Productos
- Clasificación automática por demanda (Alta, Media, Baja rotación)
- Análisis con algoritmo K-Means clustering
- Scoring ponderado de productos
- Métricas de ventas totales y frecuencia

### ⚙️ Sistema de IA Configurable
- Entrenamiento y re-entrenamiento de modelos
- Persistencia automática de modelos entrenados
- Monitoreo de estado del sistema
- Validación y métricas de rendimiento

## 🏗️ Estructura del Proyecto

```
python-ai-api/
├── app.py                          # API principal con Flask
├── models/
│   ├── __init__.py
│   ├── sales_predictor.py          # Modelo de predicción de ventas
│   ├── product_classifier.py       # Clasificador de productos
│   └── saved/                      # Modelos entrenados (auto-generado)
├── utils/
│   ├── __init__.py
│   └── data_processor.py           # Procesador de datos
├── requirements.txt                # Dependencias Python
├── run.py                         # Script de ejecución
└── README.md                      # Este archivo
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### 1. Clonar/Crear el Proyecto
```bash
mkdir python-ai-api
cd python-ai-api
```

### 2. Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install Flask==2.3.3 Flask-CORS==4.0.0 pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 joblib==1.3.2
```

O usando requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la API
```bash
python app.py
```

La API estará disponible en: `http://localhost:5000`

## 🔗 Endpoints de la API

### 1. Estado del Sistema
```http
GET /api/health
```
**Respuesta:**
```json
{
  "status": "healthy",
  "timestamp": "2024-06-05T10:30:00",
  "models": {
    "sales_predictor": "loaded",
    "product_classifier": "loaded"
  },
  "version": "1.0.0"
}
```

### 2. Predicción de Ventas
```http
POST /api/predict-sales
Content-Type: application/json

{
  "producto_id": 1,
  "producto_nombre": "Producto A",
  "dias_predecir": 30,
  "temporada": "alta",
  "datos_historicos": [
    {
      "fecha": "2024-01-01",
      "cantidad": 10,
      "precio": 100.0
    }
  ]
}
```

**Respuesta:**
```json
{
  "success": true,
  "prediccion": [25.5, 26.2, 24.8],
  "confianza": 0.85,
  "tendencia": "creciente",
  "recomendaciones": [
    "Incrementar inventario en 20%",
    "Monitorear stock para próximas 2 semanas"
  ]
}
```

### 3. Clasificación de Productos
```http
POST /api/classify-products
Content-Type: application/json

{
  "productos": [
    {
      "producto_id": 1,
      "producto_nombre": "Producto A",
      "ventas_totales": 1000,
      "frecuencia_ventas": 50
    }
  ]
}
```

**Respuesta:**
```json
{
  "success": true,
  "clasificaciones": [
    {
      "producto_id": 1,
      "producto_nombre": "Producto A",
      "categoria_demanda": "Alta",
      "score": 8.5,
      "recomendacion": "Producto estrella - mantener stock alto"
    }
  ],
  "resumen": {
    "alta_demanda": 15,
    "media_demanda": 25,
    "baja_demanda": 10
  }
}
```

### 4. Entrenar Modelos
```http
POST /api/train-model
Content-Type: application/json

{
  "modelo": "sales_predictor",
  "datos_entrenamiento": [...],
  "validar": true
}
```

## 🔧 Integración con Aplicación .NET

### Configuración en Blazor
```csharp
private string urlApiPython = "http://localhost:5000";

// Ejemplo de uso en tu página de reportes
private async Task<PredictionResult> PredecirVentas(PredictionRequest request)
{
    var json = JsonSerializer.Serialize(request);
    var content = new StringContent(json, Encoding.UTF8, "application/json");
    
    var response = await _httpClient.PostAsync($"{urlApiPython}/api/predict-sales", content);
    var result = await response.Content.ReadAsStringAsync();
    
    return JsonSerializer.Deserialize<PredictionResult>(result);
}
```

## 🛠️ Desarrollo y Personalización

### Agregar Nuevos Modelos
1. Crear nuevo archivo en `models/`
2. Implementar clase base con métodos `train()` y `predict()`
3. Registrar en `app.py`

### Modificar Algoritmos
- **Sales Predictor**: Editar `models/sales_predictor.py`
- **Product Classifier**: Editar `models/product_classifier.py`

### Agregar Nuevas Características
- Modificar `utils/data_processor.py` para preprocesamiento
- Actualizar endpoints en `app.py`

## 📊 Monitoreo y Logs

Los logs se guardan automáticamente y incluyen:
- Timestamps de requests
- Errores de predicción
- Métricas de rendimiento
- Estado de modelos

```bash
# Ver logs en tiempo real
tail -f logs/api.log
```

## 🔒 Configuración de Producción

### Variables de Entorno
```bash
export FLASK_ENV=production
export API_PORT=5000
export MODEL_PATH=/path/to/models
export LOG_LEVEL=INFO
```

### Docker (Opcional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🧪 Testing

### Probar Endpoints
```bash
# Instalar requests para testing
pip install requests

# Ejecutar tests
python -m pytest tests/
```

### Ejemplo de Test Manual
```python
import requests

# Test de salud
response = requests.get('http://localhost:5000/api/health')
print(response.json())
```

## 📈 Métricas y Rendimiento

### Métricas del Modelo
- **Accuracy**: Precisión de clasificación
- **MAE**: Error medio absoluto para predicciones
- **R²**: Coeficiente de determinación
- **Tiempo de respuesta**: < 500ms por request

### Optimización
- Caché de modelos en memoria
- Procesamiento asíncrono para datos grandes
- Validación de entrada optimizada

## 🆘 Solución de Problemas

### Errores Comunes

**Error: "Module not found"**
```bash
# Verificar entorno virtual activado
pip list
pip install -r requirements.txt
```

**Error: "Port already in use"**
```bash
# Cambiar puerto en app.py
app.run(host='0.0.0.0', port=5001, debug=True)
```

**Error: "Model not found"**
```bash
# Entrenar modelos iniciales
curl -X POST http://localhost:5000/api/train-model
```

## 📞 Soporte

Para reportar bugs o solicitar características:
1. Verificar logs de error
2. Reproducir el problema
3. Documentar pasos para reproducir

## 📝 Changelog

### v1.0.0 (2024-06-05)
- ✅ Implementación inicial de API REST
- ✅ Modelos de predicción de ventas
- ✅ Clasificación de productos por demanda
- ✅ Sistema de entrenamiento automático
- ✅ Integración con aplicaciones .NET

---

**Desarrollado para integración con sistemas de reportes y análisis de ventas** 🚀
