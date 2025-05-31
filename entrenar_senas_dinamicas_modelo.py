import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

DATA_DIR = "senas_dinamicas"
GESTOS = sorted(os.listdir(DATA_DIR))
SECUENCIA_FRAMES = 30

X, y = [], []

print("📦 Cargando secuencias...")

for label, gesto in enumerate(GESTOS):
    folder = os.path.join(DATA_DIR, gesto)
    for archivo in os.listdir(folder):
        if archivo.endswith(".npy"):
            datos = np.load(os.path.join(folder, archivo))
            if datos.shape == (SECUENCIA_FRAMES, 63):
                X.append(datos)
                y.append(label)

X = np.array(X)
y = to_categorical(y)

# === DIVIDIR DATOS ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODELO LSTM ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SECUENCIA_FRAMES, 63)),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(len(GESTOS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# === GUARDAR MODELO ===
model.save("modelo/lstm_senas_dinamicas.h5")
with open("modelo/senas_dinamicas_labels.pkl", "wb") as f:
    pickle.dump(GESTOS, f)

print("✅ Modelo LSTM guardado.")