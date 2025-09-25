# TAREA 1: Aplicación de RNNs al Modelamiento de Lenguaje en Español (LSTM/BiLSTM)

**Curso:** Fundamentos de Computación Inteligente  
**Objetivo:** Experimentar con modelos de Redes Neuronales Recurrentes (RNNs), específicamente LSTM y BiLSTM, para el modelado del lenguaje en español.

## 📋 Componentes Implementados

✅ **1. Carga del Dataset** - `spanish_billion_words_clean` de Hugging Face  
✅ **2. Tokenización y Creación del Vocabulario** - Mapeo palabra↔índice  
✅ **3. Creación del Conjunto de Entrenamiento (X, Y)** - Secuencias de entrada y siguiente palabra  
✅ **4. Padding y Truncado** - MAX_LEN ≤ 50 como especificado  
✅ **5. División del Conjunto** - Train/Test 80%/20% con `train_test_split`  
✅ **6. Construcción del Modelo LSTM/BiLSTM** - Arquitectura configurable  
✅ **7. Entrenamiento con Early Stopping** - Prevención de overfitting  
✅ **8. Cálculo de la Perplejidad** - Métrica de evaluación completa  
✅ **9. Predicción de la Próxima Palabra** - Función `predict_next_word`  
✅ **10. Uso de sparse_categorical_crossentropy** - Sin one-hot encoding  

## 📁 Archivos del Proyecto

### Archivos Principales
- **`Tarea1_ModelamientoLenguaje_LSTM_BiLSTM.ipynb`** - 📓 Notebook principal con documentación completa
- **`modelamiento_lenguaje-BiLSTM.py`** - 🐍 Script principal con implementación completa
- **`modelamiento_lenguaje-LSTMs copy.py`** - 🐍 Versión LSTM actualizada con mejoras
- **`demo_completo.py`** - 🎯 Demo interactivo que muestra todas las funcionalidades

### Documentación
- **`README.md`** - 📖 Este archivo de documentación
- **`Tarea 1 -modelamiento-del lenguaje.pdf`** - 📄 Especificaciones originales del proyecto

## 🚀 Cómo Ejecutar

### Opción 1: Jupyter Notebook (Recomendado)
```bash
jupyter notebook Tarea1_ModelamientoLenguaje_LSTM_BiLSTM.ipynb
```

### Opción 2: Script Python Completo
```bash
python3 modelamiento_lenguaje-BiLSTM.py
```

### Opción 3: Demo Rápido
```bash
python3 demo_completo.py
```

## ⚙️ Configuración de Parámetros

Los parámetros principales pueden modificarse en la sección de configuración:

```python
# Configuración del modelo
USE_BIDIRECTIONAL = True   # True: BiLSTM, False: LSTM
EMBEDDING_DIM = 50         # Dimensión del embedding (100-300 recomendado)
LSTM_UNITS = 64            # Unidades LSTM (64-128 recomendado)
DENSE_UNITS = 64           # Unidades capa densa
EPOCHS = 30                # Número de épocas
BATCH_SIZE = 8             # Tamaño del batch
LEARNING_RATE = 0.001      # Tasa de aprendizaje
```

## 🏗️ Arquitectura del Modelo

### BiLSTM (USE_BIDIRECTIONAL = True)
```
Input → Embedding(50D) → Bidirectional LSTM(64) → Dense(64, ReLU) → Dense(vocab_size, Softmax)
```

### LSTM (USE_BIDIRECTIONAL = False)
```
Input → Embedding(50D) → LSTM(64) → Dense(64, ReLU) → Dense(vocab_size, Softmax)
```

## 📊 Métricas de Evaluación

### Perplejidad
La perplejidad se calcula usando la fórmula:
```
perplexity = exp(-average_log_likelihood)
```

**Tabla de Interpretación:**
- < 10: Excelente
- 10-50: Muy bueno
- 50-100: Bueno
- 100-200: Aceptable
- 200-500: Regular
- \> 500: Pobre

### Otras Métricas
- **Loss:** sparse_categorical_crossentropy
- **Accuracy:** Precisión en predicción de siguiente palabra
- **Validation Loss/Accuracy:** Métricas en conjunto de validación

## 🔧 Características Técnicas

### Preprocesamiento de Datos
- **Tokenización:** Vocabulary único con token `<OOV>` para palabras desconocidas
- **Secuencias:** Para cada oración [w1, w2, ..., wn], genera pares (w1) → w2, (w1, w2) → w3, etc.
- **Padding:** Pre-padding con truncado para MAX_LEN = 50
- **Split:** 80% entrenamiento, 20% prueba con `train_test_split`

### Optimizaciones
- **Early Stopping:** Monitoreo de `val_loss` con patience=5
- **Adam Optimizer:** Tasa de aprendizaje configurable
- **Sparse Categorical Crossentropy:** Sin necesidad de one-hot encoding
- **Restoration:** Mejores pesos restaurados automáticamente

## 🎯 Ejemplos de Uso

### Predicción de Siguiente Palabra
```python
test_cases = [
    "el gato se sentó en la",
    "los estudiantes abrieron sus", 
    "la maestra escribió en el"
]

for sentence in test_cases:
    next_word = predict_next_word(model, tokenizer, sentence, MAX_LEN)
    print(f"'{sentence}' → '{next_word}'")
```

### Generación de Texto
```python
generated = generate_text(model, tokenizer, "el perro", MAX_LEN, num_words_to_generate=8)
print(f"Texto generado: '{generated}'")
```

### Predicciones Top-K
```python
top_predictions = predict_next_word(model, tokenizer, "el gato", MAX_LEN, top_k=3)
for i, (word, prob) in enumerate(top_predictions, 1):
    print(f"{i}. '{word}' (probabilidad: {prob:.4f})")
```

## 📋 Requisitos

### Dependencias Python
```bash
pip install tensorflow datasets numpy scikit-learn matplotlib
```

### Versiones Probadas
- Python 3.8+
- TensorFlow 2.20.0
- datasets 4.1.1
- numpy 2.3.3
- scikit-learn 1.7.2

## 🏆 Resultados de Ejemplo

Con dataset de 50,000 muestras:
- **Vocabulario:** ~15,000-30,000 palabras
- **Secuencias de entrenamiento:** ~400,000-800,000
- **Perplejidad típica:** 50-200 (Bueno-Aceptable)
- **Tiempo de entrenamiento:** 30-60 minutos (CPU)

## 🔄 Variantes del Modelo

### Para experimentar con diferentes configuraciones:

**Modelo Pequeño (Rápido):**
```python
EMBEDDING_DIM = 32
LSTM_UNITS = 32
DENSE_UNITS = 32
EPOCHS = 10
```

**Modelo Grande (Mejor rendimiento):**
```python
EMBEDDING_DIM = 300
LSTM_UNITS = 128
DENSE_UNITS = 128
EPOCHS = 50
```

**Modelo BiLSTM vs LSTM:**
```python
USE_BIDIRECTIONAL = True   # BiLSTM (más parámetros, mejor contexto)
USE_BIDIRECTIONAL = False  # LSTM (más rápido, menos parámetros)
```

## 🚨 Solución de Problemas

### Error de Conectividad con Hugging Face
Si hay problemas de red, el código automáticamente utiliza oraciones locales:
```python
# Fallback automático a dataset local si falla la conexión
sentences = [
    "el gato se sentó en la alfombra",
    "el perro corrió en el parque",
    # ... más oraciones locales
]
```

### Memoria Insuficiente
Reducir el tamaño del dataset:
```python
DATASET_TAKE = 10000  # Reducir de 50000 a 10000
BATCH_SIZE = 4        # Reducir batch size
```

### Entrenamiento Muy Lento
Usar menos épocas y modelo más pequeño:
```python
EPOCHS = 10
EMBEDDING_DIM = 32
LSTM_UNITS = 32
```

## 🎓 Entregables

1. **Archivo .ipynb** ✅ - `Tarea1_ModelamientoLenguaje_LSTM_BiLSTM.ipynb`
2. **Documentación completa** ✅ - Cada paso explicado en el notebook
3. **Código funcional** ✅ - Probado y validado
4. **Todos los componentes requeridos** ✅ - Lista completa implementada

## 👥 Notas para el Grupo

- **Trabajo en equipo:** Máximo 3 estudiantes
- **Plataformas:** Jupyter AI, Colab o local (32GB RAM disponible)
- **Formato:** Archivo .ipynb bien documentado
- **Plazo:** Dos semanas desde asignación

---

**¡Proyecto completado exitosamente!** 🎉

Para cualquier duda o modificación, consultar el código fuente o ejecutar el demo interactivo.