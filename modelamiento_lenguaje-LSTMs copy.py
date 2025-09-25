import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import math

# ----------------------------
# 1. Frases en español (dataset)
# ----------------------------

print("Cargando dataset...")
try:
    dataset = load_dataset("jhonparra18/spanish_billion_words_clean", split="train", streaming=True).take(50000)
    sentences = [example['text'] for example in dataset if len(example['text'].split()) > 3]
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using local Spanish sentences for demonstration...")
    sentences = [
        "el gato se sentó en la alfombra",
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
        "la ciudad es muy grande",
        "el agua está muy fría hoy"
    ]

print(f"Total de oraciones cargadas: {len(sentences)}")

# ----------------------------
# 2. Tokenización
# ----------------------------
tokenizer = Tokenizer(oov_token="<OOV>")  # <OOV> para palabras desconocidas
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

# Longitud máxima de las secuencias (con restricción MAX_LEN <= 50)
raw_max_length = max([len(seq) for seq in X])
MAX_LEN = min(50, raw_max_length)  # Aplicar restricción de máximo 50
print(f"\nLongitud máxima encontrada: {raw_max_length}")
print(f"MAX_LEN aplicado (truncado a 50): {MAX_LEN}")

# ----------------------------
# 4. Aplicar padding y truncado a las secuencias de entrada
# ----------------------------
X_padded = pad_sequences(X, maxlen=MAX_LEN, padding='pre', truncating='pre')
y = np.array(y)  # Etiquetas: ya es un array 1D

print(f"Forma de X_padded: {X_padded.shape}")  # (número de muestras, MAX_LEN)
print(f"Forma de y: {y.shape}")
print("Ejemplo de X_padded:")
print(X_padded[:5])
print("Ejemplo de y:")
print(y[:5])

# ----------------------------
# 5. División del conjunto (Train/Test: 80%/20%)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

print(f"\nDivisión del conjunto:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Nota: Usamos sparse_categorical_crossentropy, NO necesitamos one-hot encoding
print("\n✅ Usando sparse_categorical_crossentropy - NO necesitamos one-hot encoding")

# ----------------------------
# 6. Construir el modelo LSTM
# ----------------------------
model = Sequential([
    Embedding(vocab_size, 50, input_length=MAX_LEN),
    LSTM(64),  # Capa LSTM
    Dense(64, activation='relu'),
    Dense(vocab_size, activation='softmax')  # Probabilidades para cada palabra
])

model.compile(
    loss='sparse_categorical_crossentropy',  # Cambio: usar sparse_categorical_crossentropy
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# 7. Entrenar el modelo con Early Stopping
# ----------------------------
# Configurar Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',     # Métrica a monitorear
    patience=5,             # Número de épocas sin mejora antes de parar
    restore_best_weights=True,  # Restaurar los mejores pesos
    verbose=1               # Mostrar información cuando se pare
)

print("\nEntrenando el modelo...")
print("Early Stopping habilitado: val_loss con patience=5")

history = model.fit(
    X_train, y_train,  # Usar conjuntos de entrenamiento divididos
    epochs=30, 
    verbose=1, 
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stopping]  # Agregar Early Stopping
)

# ----------------------------
# 8. Función para predecir la palabra siguiente
# ----------------------------
def predict_next_word(model, tokenizer, sentence, max_length):
    """
    Predice la palabra siguiente dada una oración en español.
    """
    # Tokenizar la oración
    sequence = tokenizer.texts_to_sequences([sentence])[0]

    if len(sequence) == 0:
        return "<no_reconocido>"

    # Aplicar padding y truncado
    sequence_padded = pad_sequences([sequence], maxlen=max_length, padding='pre', truncating='pre')

    # Predecir
    prediction = model.predict(sequence_padded, verbose=0)
    predicted_idx = np.argmax(prediction[0])

    # Obtener palabra desde el índice
    predicted_word = tokenizer.index_word.get(predicted_idx, "<desconocido>")

    return predicted_word

# ----------------------------
# 9. Cálculo de la Perplejidad
# ----------------------------
def calculate_perplexity(model, X_test, y_test):
    """
    Calcula la perplejidad del modelo en el conjunto de prueba.
    
    Fórmula: perplexity = exp(-average_log_likelihood)
    """
    # Obtener probabilidades predichas para el conjunto de prueba
    predictions = model.predict(X_test, verbose=0)
    
    # Calcular log-likelihood promedio
    log_likelihoods = []
    for i in range(len(y_test)):
        true_word_idx = y_test[i]
        predicted_prob = predictions[i][true_word_idx]
        # Evitar log(0) agregando un pequeño epsilon
        predicted_prob = max(predicted_prob, 1e-10)
        log_likelihoods.append(np.log(predicted_prob))
    
    # Calcular perplejidad
    average_log_likelihood = np.mean(log_likelihoods)
    perplexity = np.exp(-average_log_likelihood)
    
    return perplexity

def interpret_perplexity(perplexity_value):
    """Interpreta el valor de perplejidad según rangos comunes."""
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
print("\n=== EVALUACIÓN EN CONJUNTO DE PRUEBA ===")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Calcular perplejidad
print("\n=== CÁLCULO DE PERPLEJIDAD ===")
perplexity = calculate_perplexity(model, X_test, y_test)
interpretation = interpret_perplexity(perplexity)

print(f"Perplejidad en conjunto de prueba: {perplexity:.2f}")
print(f"Interpretación: {interpretation}")

# Tabla de interpretación
print("\n=== TABLA DE INTERPRETACIÓN DE PERPLEJIDAD ===")
print("< 10:     Excelente")
print("10-50:    Muy bueno") 
print("50-100:   Bueno")
print("100-200:  Aceptable")
print("200-500:  Regular")
print("> 500:    Pobre")

# ----------------------------
# 10. Pruebas del modelo
# ----------------------------
test_cases = [
    "el gato se sentó en la",
    "los estudiantes abrieron sus",
    "la maestra escribió en el",
    "el niño jugó con un",
    "el pájaro voló sobre el",
    "el sol se elevó en el"
]

print("\n--- Predicciones en español ---")
for sentence in test_cases:
    next_word = predict_next_word(model, tokenizer, sentence, MAX_LEN)
    print(f"{sentence} → {next_word}")

print(f"\n=== RESUMEN FINAL ===")
print(f"Vocabulario: {vocab_size:,} palabras")
print(f"Secuencias de entrenamiento: {len(X_train):,}")
print(f"Secuencias de prueba: {len(X_test):,}")
print(f"MAX_LEN aplicado: {MAX_LEN}")
print(f"Perplejidad: {perplexity:.2f} ({interpretation})")
print("✅ Todos los requerimientos implementados correctamente!")