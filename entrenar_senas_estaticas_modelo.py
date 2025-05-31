# === entrenar_modelo.py ===
# Este script carga los vectores guardados de cada letra y entrena un modelo
# de reconocimiento usando KNN (K-Nearest Neighbors). Guarda el modelo en un archivo .pkl.

import os
import numpy as np
import pickle  # Para guardar el modelo entrenado
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# === CONFIGURACIÓN ===
DATA_DIR = "senas_estaticas"  # Carpeta con los archivos .npy
MODEL_PATH = "modelo/knn_senas_estaticas.pkl"
letters = "abcdefghijklmnopqrstuvwxyz"

X, y = [], []  # X = vectores, y = etiquetas (números)

print("📦 Cargando datos de landmarks...")

# === LEER TODOS LOS ARCHIVOS .NPY POR CADA LETRA ===
for idx, letter in enumerate(letters):
    folder = os.path.join(DATA_DIR, letter)
    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            data = np.load(path)
            if data.shape == (63,):  # 21 landmarks × 3 coordenadas
                X.append(data)
                y.append(idx)  # Etiqueta numérica (0=a, 1=b, ..., 25=z)

# === VALIDAR QUE HAYA DATOS ===
if not X:
    raise RuntimeError("❌ No se encontraron vectores válidos en 'senas_estaticas/'")

X = np.array(X)
y = np.array(y)

# === DIVIDIR EN ENTRENAMIENTO Y PRUEBA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === ENTRENAR MODELO KNN ===
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# === GUARDAR MODELO ENTRENADO ===
os.makedirs("modelo", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(knn, f)

print(f"✅ Modelo entrenado y guardado en: {MODEL_PATH}")
print(f"📊 Precisión en entrenamiento: {knn.score(X_train, y_train):.2f}")
print(f"📊 Precisión en prueba: {knn.score(X_test, y_test):.2f}")