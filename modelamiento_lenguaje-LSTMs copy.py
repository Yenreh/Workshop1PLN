import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
import math

# ----------------------------
# 1. Frases en español (dataset pequeño)
# ----------------------------

print("Cargando dataset...")
dataset = load_dataset("jhonparra18/spanish_billion_words_clean", split="train", streaming=True).take(50000)
sentences = [example['text'] for example in dataset if len(example['text'].split()) > 3]

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

# Longitud máxima de las secuencias
max_length = max([len(seq) for seq in X])
print(f"\nLongitud máxima de secuencia: {max_length}")
print('máxima longitud de sentencia', max_length)

# ----------------------------
# 4. Aplicar padding a las secuencias de entrada
# ----------------------------
X_padded = pad_sequences(X, maxlen=max_length, padding='pre')
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
# 6. Construir el modelo LSTM
# ----------------------------
model = Sequential([
    Embedding(vocab_size, 50, input_length=max_length),
    LSTM(64),  # Capa LSTM
    Dense(64, activation='relu'),
    Dense(vocab_size, activation='softmax')  # Probabilidades para cada palabra
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# 7. Entrenar el modelo
# ----------------------------
print("\nEntrenando el modelo...")
history = model.fit(X_padded, y_onehot, epochs=30, verbose=1, batch_size=8)

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

    # Padrar secuencia
    sequence_padded = pad_sequences([sequence], maxlen=max_length, padding='pre')

    # Predecir
    prediction = model.predict(sequence_padded, verbose=0)
    predicted_idx = np.argmax(prediction[0])

    # Obtener palabra desde el índice
    predicted_word = tokenizer.index_word.get(predicted_idx, "<desconocido>")

    return predicted_word

# ----------------------------
# 9. Pruebas del modelo
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
    next_word = predict_next_word(model, tokenizer, sentence, max_length)
    print(f"{sentence} → {next_word}")