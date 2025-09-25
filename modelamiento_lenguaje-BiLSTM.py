"""
TAREA 1: MODELAMIENTO DEL LENGUAJE CON BiLSTM
============================================

Este script implementa un modelo de lenguaje bidireccional (BiLSTM) para la predicción 
de la siguiente palabra en secuencias de texto en español.

CARACTERÍSTICAS PRINCIPALES:
- Arquitectura BiLSTM (Bidirectional LSTM) configurable
- Parámetros completamente parametrizados para fácil experimentación
- Funciones avanzadas de predicción y generación de texto
- Evaluación completa del modelo con métricas detalladas
- Uso del dataset "spanish_billion_words_clean" de HuggingFace

CONFIGURACIÓN:
- Modifica las constantes en la sección "PARÁMETROS GLOBALES" para experimentar
- Cambia USE_BIDIRECTIONAL=False para usar LSTM unidireccional
- Ajusta DATASET_TAKE para cambiar el tamaño del dataset
- Personaliza la arquitectura modificando EMBEDDING_DIM, LSTM_UNITS, etc.

AUTOR: [Tu nombre]
FECHA: 2025
UNIVERSIDAD: [Tu universidad]
CURSO: Procesamiento de Lenguaje Natural (PLN) - Workshop 1
"""

# ----------------------------
# 0. PARÁMETROS GLOBALES CONFIGURABLES
# ----------------------------

# Dataset configuration
DATASET_NAME = "jhonparra18/spanish_billion_words_clean"
DATASET_SPLIT = "train"
DATASET_STREAMING = True
DATASET_TAKE = 50000  # Número de ejemplos a tomar del dataset
MIN_WORDS_PER_SENTENCE = 4  # Filtro: oraciones con al menos N palabras
OOV_TOKEN = "<OOV>"  # Token para palabras fuera del vocabulario

# Model architecture parameters
EMBEDDING_DIM = 50  # Dimensión del embedding de palabras
LSTM_UNITS = 64  # Número de unidades en las capas LSTM
DENSE_UNITS = 64  # Número de unidades en la capa densa intermedia
USE_BIDIRECTIONAL = False  # Usar BiLSTM en lugar de LSTM unidireccional

# Training parameters
EPOCHS = 30  # Número de épocas de entrenamiento
BATCH_SIZE = 8  # Tamaño del batch
VALIDATION_SPLIT = 0.2  # Porcentaje de datos para validación
LEARNING_RATE = 0.001  # Tasa de aprendizaje

# Sequence processing
PADDING_TYPE = 'pre'  # Tipo de padding: 'pre' o 'post'

# Output parameters
VERBOSE_TRAINING = 1  # Nivel de verbosidad durante entrenamiento
VERBOSE_PREDICTION = 0  # Nivel de verbosidad durante predicción

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import math

# ----------------------------
# 1. Frases en español (dataset)
# ----------------------------

print("Cargando dataset...")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=DATASET_STREAMING).take(DATASET_TAKE)
sentences = [example['text'] for example in dataset if len(example['text'].split()) >= MIN_WORDS_PER_SENTENCE]

# ▲ Usa un submuestreo si tienes poca RAM (opcional)
# dataset = dataset.select(range(500))  # Ajusta según tu hardware
# Extraer textos y filtrar oraciones muy cortas
# sentences = [example['text'] for example in dataset if len(example['text'].split()) > 3]
print(f"Total de oraciones cargadas: {len(sentences)}")

# sentences = [
#     "el gato se sentó en la alfombra",
#     "el perro corrió en el parque",
#     "los estudiantes abrieron sus libros",
#     "la maestra escribió en el pizarrón",
#     "el niño jugó con un juguete",
#     "el coche condujo por la carretera",
#     "el pájaro voló sobre el árbol",
#     "el sol se elevó en el cielo"
# ]

# ----------------------------
# 2. Tokenización
# ----------------------------
tokenizer = Tokenizer(oov_token=OOV_TOKEN)  # Token para palabras desconocidas
tokenizer.fit_on_texts(sentences)

# Convertir frases a secuencias numéricas
sequences = tokenizer.texts_to_sequences(sentences)

# Tamaño del vocabulario
vocab_size = len(tokenizer.word_index) + 1  # +1 por el padding (índice 0)
print(f"Vocabulario: {vocab_size} palabras")
print("Vocabulario (palabra → índice):")
print(tokenizer.word_index)

# ----------------------------
# 3. Crear secuencias de entrada y palabra siguiente
# ----------------------------
X = []  # Secuencias de entrada: [palabra1], [palabra1, palabra2], ...
y = []  # Palabra siguiente (objetivo)

for seq in sequences:
    for i in range(len(seq) - 1):
        X.append(seq[:i+1])    # Desde el inicio hasta la palabra actual
        y.append(seq[i+1])     # La siguiente palabra

# Longitud máxima de las secuencias
max_length = max([len(seq) for seq in X])
print(f"\nLongitud máxima de secuencia: {max_length}")
print('máxima longitud de sentencia', max_length)

# ----------------------------
# 4. Aplicar padding a las secuencias de entrada
# ----------------------------
X_padded = pad_sequences(X, maxlen=max_length, padding=PADDING_TYPE)
y = np.array(y)  # Etiquetas: ya es un array 1D

print(f"Forma de X_padded: {X_padded.shape}")  # (número de muestras, max_length)
print(f"Forma de y: {y.shape}")
print("Ejemplo de X_padded:")
print(X_padded[:5])
print("Ejemplo de y:")
print(y[:5])

# ----------------------------
# 5. Codificar salida como one-hot
# ----------------------------
y_onehot = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
print(f"Forma de y_onehot: {y_onehot.shape}")
print(y_onehot)
# ----------------------------
# 6. Construir el modelo BiLSTM
# ----------------------------
def create_model(vocab_size, max_length):
    """
    Crea un modelo de lenguaje usando BiLSTM o LSTM según configuración.
    
    Args:
        vocab_size (int): Tamaño del vocabulario
        max_length (int): Longitud máxima de las secuencias
    
    Returns:
        tensorflow.keras.Model: Modelo compilado
    """
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length, name='embedding'),
        # Usar BiLSTM si está habilitado, sino LSTM unidireccional
        Bidirectional(LSTM(LSTM_UNITS, name='lstm'), name='bidirectional_lstm') if USE_BIDIRECTIONAL 
        else LSTM(LSTM_UNITS, name='lstm'),
        Dense(DENSE_UNITS, activation='relu', name='dense_hidden'),
        Dense(vocab_size, activation='softmax', name='output')  # Probabilidades para cada palabra
    ])
    
    # Configurar optimizador con tasa de aprendizaje personalizada
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

model = create_model(vocab_size, max_length)

model.summary()

# ----------------------------
# 7. Entrenar el modelo
# ----------------------------
print(f"\nEntrenando el modelo {'BiLSTM' if USE_BIDIRECTIONAL else 'LSTM'}...")
print(f"Arquitectura: {EMBEDDING_DIM}D embedding → {'Bi-' if USE_BIDIRECTIONAL else ''}LSTM({LSTM_UNITS}) → Dense({DENSE_UNITS}) → Softmax({vocab_size})")
print(f"Parámetros de entrenamiento: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")

history = model.fit(
    X_padded, y_onehot,
    epochs=EPOCHS,
    verbose=VERBOSE_TRAINING,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    shuffle=True
)

# ----------------------------
# 8. Función para predecir la palabra siguiente
# ----------------------------
def predict_next_word(model, tokenizer, sentence, max_length, top_k=1):
    """
    Predice la(s) palabra(s) siguiente(s) dada una oración en español.
    
    Args:
        model: Modelo entrenado
        tokenizer: Tokenizador ajustado
        sentence (str): Oración de entrada
        max_length (int): Longitud máxima de secuencia
        top_k (int): Número de predicciones top a retornar
    
    Returns:
        str o list: Palabra predicha (top_k=1) o lista de predicciones (top_k>1)
    """
    # Tokenizar la oración
    sequence = tokenizer.texts_to_sequences([sentence])[0]

    if len(sequence) == 0:
        return "<no_reconocido>" if top_k == 1 else ["<no_reconocido>"]

    # Aplicar padding usando el mismo tipo configurado
    sequence_padded = pad_sequences([sequence], maxlen=max_length, padding=PADDING_TYPE)

    # Predecir probabilidades
    prediction = model.predict(sequence_padded, verbose=VERBOSE_PREDICTION)
    
    if top_k == 1:
        # Retornar solo la predicción más probable
        predicted_idx = np.argmax(prediction[0])
        predicted_word = tokenizer.index_word.get(predicted_idx, "<desconocido>")
        return predicted_word
    else:
        # Retornar las top_k predicciones más probables
        top_indices = np.argsort(prediction[0])[-top_k:][::-1]
        predictions = []
        for idx in top_indices:
            word = tokenizer.index_word.get(idx, "<desconocido>")
            prob = prediction[0][idx]
            predictions.append((word, prob))
        return predictions

def generate_text(model, tokenizer, seed_text, max_length, num_words_to_generate=10):
    """
    Genera texto continuando desde un texto semilla.
    
    Args:
        model: Modelo entrenado
        tokenizer: Tokenizador ajustado
        seed_text (str): Texto inicial
        max_length (int): Longitud máxima de secuencia
        num_words_to_generate (int): Número de palabras a generar
    
    Returns:
        str: Texto generado
    """
    generated_text = seed_text
    
    for _ in range(num_words_to_generate):
        next_word = predict_next_word(model, tokenizer, generated_text, max_length)
        if next_word in ["<no_reconocido>", "<desconocido>"]:
            break
        generated_text += " " + next_word
    
    return generated_text

# ----------------------------
# 9. Evaluación y pruebas del modelo
# ----------------------------

def evaluate_model_performance(history):
    """
    Muestra métricas de rendimiento del modelo durante el entrenamiento.
    """
    print("\n=== MÉTRICAS DE ENTRENAMIENTO ===")
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    
    if 'val_loss' in history.history:
        final_val_loss = history.history['val_loss'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        print(f"Loss final: {final_loss:.4f} | Val Loss: {final_val_loss:.4f}")
        print(f"Accuracy final: {final_accuracy:.4f} | Val Accuracy: {final_val_accuracy:.4f}")
    else:
        print(f"Loss final: {final_loss:.4f}")
        print(f"Accuracy final: {final_accuracy:.4f}")

# Evaluar rendimiento
evaluate_model_performance(history)

# Casos de prueba para predicción de siguiente palabra
test_cases = [
    "el gato se sentó en la",
    "los estudiantes abrieron sus",
    "la maestra escribió en el",
    "el niño jugó con un",
    "el pájaro voló sobre el",
    "el sol se elevó en el",
    "la casa tiene una",
    "me gusta comer",
    "vamos a la"
]

print(f"\n=== PREDICCIÓN DE SIGUIENTE PALABRA ({'BiLSTM' if USE_BIDIRECTIONAL else 'LSTM'}) ===")
for sentence in test_cases:
    next_word = predict_next_word(model, tokenizer, sentence, max_length)
    print(f"'{sentence}' → '{next_word}'")

# Predicciones top-k para algunos casos
print(f"\n=== TOP-3 PREDICCIONES ===")
for sentence in test_cases[:3]:
    top_predictions = predict_next_word(model, tokenizer, sentence, max_length, top_k=3)
    print(f"'{sentence}':")
    for i, (word, prob) in enumerate(top_predictions, 1):
        print(f"  {i}. '{word}' (probabilidad: {prob:.4f})")
    print()

# Generación de texto
print(f"=== GENERACIÓN DE TEXTO ===")
seed_texts = [
    "el perro",
    "los estudiantes",
    "en la casa"
]

for seed in seed_texts:
    generated = generate_text(model, tokenizer, seed, max_length, num_words_to_generate=8)
    print(f"Semilla: '{seed}'")
    print(f"Generado: '{generated}'")
    print()

print("=== RESUMEN DEL MODELO ===")
print(f"Vocabulario: {vocab_size:,} palabras")
print(f"Secuencias de entrenamiento: {len(X_padded):,}")
print(f"Longitud máxima de secuencia: {max_length}")
print(f"Arquitectura: {'BiLSTM' if USE_BIDIRECTIONAL else 'LSTM'}")
print(f"Parámetros del modelo: {model.count_params():,}")
print(f"Dataset utilizado: {DATASET_NAME} (primeras {DATASET_TAKE:,} muestras)")
print("\n¡Entrenamiento completado!")