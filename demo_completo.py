#!/usr/bin/env python3
"""
DEMO COMPLETO - TAREA 1: MODELAMIENTO DEL LENGUAJE CON LSTM/BiLSTM
================================================================

Este script demuestra todas las funcionalidades implementadas para la Tarea 1.
Ejecuta un ejemplo completo con un dataset pequeño para mostrar todas las características.

COMPONENTES DEMOSTRADOS:
✅ Carga del Dataset (con fallback a datos locales)
✅ Tokenización y Creación del Vocabulario  
✅ Creación del Conjunto de Entrenamiento (X, Y)
✅ Padding y Truncado (MAX_LEN ≤ 50)
✅ División del Conjunto (Train/Test 80%/20%)
✅ Construcción del Modelo LSTM/BiLSTM
✅ Entrenamiento con Early Stopping
✅ Cálculo de la Perplejidad
✅ Predicción de la Próxima Palabra
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Configuración del modelo
USE_BIDIRECTIONAL = True  # Cambiar a False para LSTM unidireccional
EMBEDDING_DIM = 32
LSTM_UNITS = 32
DENSE_UNITS = 32
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.001

print("=" * 60)
print("DEMO: MODELAMIENTO DEL LENGUAJE CON LSTM/BiLSTM")
print("=" * 60)

# Dataset de demostración en español
demo_sentences = [
    "el gato se sentó en la alfombra roja",
    "el perro corrió en el parque grande",
    "los estudiantes abrieron sus libros nuevos",
    "la maestra escribió en el pizarrón blanco",
    "el niño jugó con un juguete rojo",
    "el coche condujo por la carretera principal",
    "el pájaro voló sobre el árbol alto",
    "el sol se elevó en el cielo azul",
    "la casa tiene una puerta grande",
    "me gusta comer comida deliciosa",
    "vamos a la playa este verano",
    "el libro está sobre la mesa",
    "los niños juegan en el jardín",
    "la ciudad es muy grande y hermosa",
    "el agua está muy fría hoy",
    "mi familia vive en una casa",
    "el profesor enseña en el aula",
    "los pájaros cantan en el árbol",
    "la luna brilla en la noche",
    "el viento sopla muy fuerte"
]

print(f"1. DATASET: {len(demo_sentences)} oraciones en español")
print(f"Ejemplo: '{demo_sentences[0]}'")

# 2. Tokenización
print("\n2. TOKENIZACIÓN Y VOCABULARIO")
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(demo_sentences)
sequences = tokenizer.texts_to_sequences(demo_sentences)
vocab_size = len(tokenizer.word_index) + 1

print(f"Vocabulario: {vocab_size} palabras")
print("Primeras 10 palabras:", list(tokenizer.word_index.items())[:10])

# 3. Crear secuencias X, Y
print("\n3. CREACIÓN DE SECUENCIAS (X, Y)")
X, y = [], []
for seq in sequences:
    for i in range(len(seq) - 1):
        X.append(seq[:i+1])
        y.append(seq[i+1])

print(f"Total de secuencias generadas: {len(X)}")

# Ejemplo de secuencias
print("Ejemplo de secuencias X → y:")
for i in range(3):
    x_words = [tokenizer.index_word.get(idx, '<UNK>') for idx in X[i]]
    y_word = tokenizer.index_word.get(y[i], '<UNK>')
    print(f"  {x_words} → {y_word}")

# 4. Aplicar MAX_LEN constraint
print("\n4. PADDING Y TRUNCADO (MAX_LEN ≤ 50)")
raw_max_length = max([len(seq) for seq in X])
MAX_LEN = min(50, raw_max_length)
print(f"Longitud máxima encontrada: {raw_max_length}")
print(f"MAX_LEN aplicado: {MAX_LEN}")

X_padded = pad_sequences(X, maxlen=MAX_LEN, padding='pre', truncating='pre')
y = np.array(y)

print(f"X_padded shape: {X_padded.shape}")
print(f"y shape: {y.shape}")

# 5. Train/Test Split
print("\n5. DIVISIÓN TRAIN/TEST (80%/20%)")
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print("✅ Usando sparse_categorical_crossentropy (sin one-hot encoding)")

# 6. Crear modelo
print(f"\n6. CONSTRUCCIÓN DEL MODELO {'BiLSTM' if USE_BIDIRECTIONAL else 'LSTM'}")

def create_model(vocab_size, max_length):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length),
        Bidirectional(LSTM(LSTM_UNITS)) if USE_BIDIRECTIONAL else LSTM(LSTM_UNITS),
        Dense(DENSE_UNITS, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )
    return model

model = create_model(vocab_size, MAX_LEN)
model.build(input_shape=(None, MAX_LEN))
print(f"Parámetros del modelo: {model.count_params():,}")

# 7. Entrenamiento con Early Stopping
print(f"\n7. ENTRENAMIENTO CON EARLY STOPPING")
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
)

print(f"Arquitectura: {EMBEDDING_DIM}D embedding → {'Bi-' if USE_BIDIRECTIONAL else ''}LSTM({LSTM_UNITS}) → Dense({DENSE_UNITS})")
print(f"Parámetros: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Cálculo de Perplejidad
print("\n8. CÁLCULO DE PERPLEJIDAD")

def calculate_perplexity(model, X_test, y_test):
    predictions = model.predict(X_test, verbose=0)
    log_likelihoods = []
    for i in range(len(y_test)):
        true_word_idx = y_test[i]
        predicted_prob = predictions[i][true_word_idx]
        predicted_prob = max(predicted_prob, 1e-10)
        log_likelihoods.append(np.log(predicted_prob))
    
    average_log_likelihood = np.mean(log_likelihoods)
    perplexity = np.exp(-average_log_likelihood)
    return perplexity

def interpret_perplexity(perplexity_value):
    if perplexity_value < 10:
        return "Excelente"
    elif perplexity_value < 50:
        return "Muy bueno"
    elif perplexity_value < 100:
        return "Bueno"
    elif perplexity_value < 200:
        return "Aceptable"
    elif perplexity_value < 500:
        return "Regular"
    else:
        return "Pobre"

# Evaluar en conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
perplexity = calculate_perplexity(model, X_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Perplejidad: {perplexity:.2f} ({interpret_perplexity(perplexity)})")

# Tabla de interpretación
print("\nTabla de interpretación de perplejidad:")
print("< 10: Excelente | 10-50: Muy bueno | 50-100: Bueno")
print("100-200: Aceptable | 200-500: Regular | > 500: Pobre")

# 9. Predicción de siguiente palabra
print("\n9. PREDICCIÓN DE LA PRÓXIMA PALABRA")

def predict_next_word(model, tokenizer, sentence, max_length):
    sequence = tokenizer.texts_to_sequences([sentence])[0]
    if len(sequence) == 0:
        return "<no_reconocido>"
    
    sequence_padded = pad_sequences([sequence], maxlen=max_length, padding='pre', truncating='pre')
    prediction = model.predict(sequence_padded, verbose=0)
    predicted_idx = np.argmax(prediction[0])
    predicted_word = tokenizer.index_word.get(predicted_idx, "<desconocido>")
    return predicted_word

# Casos de prueba
test_cases = [
    "el gato se sentó en la",
    "los estudiantes abrieron sus",
    "la maestra escribió en el",
    "el niño jugó con un",
    "me gusta comer",
    "vamos a la"
]

print(f"Predicciones con modelo {'BiLSTM' if USE_BIDIRECTIONAL else 'LSTM'}:")
for sentence in test_cases:
    next_word = predict_next_word(model, tokenizer, sentence, MAX_LEN)
    print(f"  '{sentence}' → '{next_word}'")

# 10. Resumen final
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"✅ Vocabulario: {vocab_size:,} palabras")
print(f"✅ Secuencias entrenamiento: {len(X_train):,}")
print(f"✅ Secuencias prueba: {len(X_test):,}")
print(f"✅ MAX_LEN: {MAX_LEN}")
print(f"✅ Arquitectura: {'BiLSTM' if USE_BIDIRECTIONAL else 'LSTM'}")
print(f"✅ Parámetros modelo: {model.count_params():,}")
print(f"✅ Perplejidad: {perplexity:.2f} ({interpret_perplexity(perplexity)})")

components = [
    "Carga del Dataset",
    "Tokenización y Vocabulario",
    "Secuencias (X, Y)",
    "MAX_LEN constraint (≤ 50)",
    "Train/Test split (80%/20%)",
    "sparse_categorical_crossentropy",
    "Early Stopping",
    "Cálculo de Perplejidad",
    "Predicción siguiente palabra",
    f"Arquitectura {'BiLSTM' if USE_BIDIRECTIONAL else 'LSTM'}"
]

print(f"\nComponentes implementados:")
for i, component in enumerate(components, 1):
    print(f"  {i:2d}. ✅ {component}")

print(f"\n🎉 DEMO COMPLETADO EXITOSAMENTE!")
print(f"Para usar BiLSTM, cambiar USE_BIDIRECTIONAL = True")
print(f"Para usar LSTM simple, cambiar USE_BIDIRECTIONAL = False")