# File: entrenar_senas_estaticas_modelo.py
# === entrenar_modelo.py ===
# Este script carga los vectores de landmarks de señas estáticas guardados para cada letra,
# entrena un modelo de reconocimiento usando KNN (K-Nearest Neighbors) y
# guarda el modelo entrenado en un archivo .pkl para su uso posterior.

import os           # Importa la librería os para interactuar con el sistema operativo (listar directorios, crear carpetas).
import numpy as np  # Importa numpy para manejo eficiente de arrays numéricos.
import pickle       # Importa pickle para serializar y deserializar objetos Python (guardar el modelo).
# Importa funciones necesarias de scikit-learn.
from sklearn.model_selection import train_test_split # Para dividir los datos en conjuntos de entrenamiento y prueba.
from sklearn.neighbors import KNeighborsClassifier # El clasificador K-Nearest Neighbors.

# === CONFIGURACIÓN ===
DATA_DIR = "senas_estaticas"  # Carpeta donde se encuentran los archivos .npy con los vectores de landmarks.
MODEL_PATH = "modelo/knn_senas_estaticas.pkl" # Ruta donde se guardará el modelo entrenado.
letters = "abcdefghijklmnopqrstuvwxyz" # Cadena con las letras que se espera encontrar como etiquetas.

X, y = [], []  # Inicializa listas vacías: X para los vectores de landmarks, y para las etiquetas numéricas.

print("📦 Cargando datos de landmarks...") # Mensaje informativo.

# === LEER TODOS LOS ARCHIVOS .NPY POR CADA LETRA ===
# Itera sobre cada letra definida en la cadena 'letters'.
for idx, letter in enumerate(letters): # 'idx' será el índice numérico (0 para 'a', 1 para 'b', etc.).
    folder = os.path.join(DATA_DIR, letter) # Construye la ruta completa a la carpeta de la letra actual (ej: senas_estaticas/a).
    if not os.path.exists(folder): # Verifica si la carpeta para la letra actual existe.
        continue # Si la carpeta no existe, salta a la siguiente letra.

    # Itera sobre los archivos dentro de la carpeta de la letra.
    for file in os.listdir(folder):
        if file.endswith(".npy"): # Verifica si el archivo termina con la extensión .npy.
            path = os.path.join(folder, file) # Construye la ruta completa al archivo .npy.
            data = np.load(path) # Carga los datos (el vector de landmarks) desde el archivo .npy.
            # Verifica que la forma de los datos cargados sea la esperada (63 elementos: 21 landmarks * 3 coordenadas).
            if data.shape == (63,):
                X.append(data) # Añade el vector de landmarks a la lista X.
                y.append(idx)  # Añade la etiqueta numérica (índice de la letra) a la lista y.

# === VALIDAR QUE HAYA DATOS ===
if not X: # Verifica si la lista X está vacía (si no se cargó ningún vector).
    # Si no se encontraron datos, lanza un error y termina el script.
    raise RuntimeError("❌ No se encontraron vectores válidos en 'senas_estaticas/'")

X = np.array(X) # Convierte la lista de vectores X a un array numpy para un manejo más eficiente.
y = np.array(y) # Convierte la lista de etiquetas y a un array numpy.

# === DIVIDIR EN ENTRENAMIENTO Y PRUEBA ===
# Divide los datos cargados en dos conjuntos: uno para entrenamiento (80%) y otro para prueba (20%).
# random_state=42 asegura que la división sea la misma cada vez que se ejecuta el script, lo que ayuda a la reproducibilidad.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === ENTRENAR MODELO KNN ===
knn = KNeighborsClassifier(n_neighbors=5) # Inicializa el clasificador KNN con K=5 (considera los 5 vecinos más cercanos).
knn.fit(X_train, y_train) # Entrena el modelo KNN usando los datos de entrenamiento (X_train y y_train).

# === GUARDAR MODELO ENTRENADO ===
os.makedirs("modelo", exist_ok=True) # Crea la carpeta "modelo" si no existe. exist_ok=True evita un error si ya existe.
with open(MODEL_PATH, "wb") as f: # Abre el archivo especificado en MODEL_PATH en modo escritura binaria ('wb').
    pickle.dump(knn, f) # Serializa el objeto del modelo KNN y lo guarda en el archivo.

print(f"✅ Modelo entrenado y guardado en: {MODEL_PATH}") # Mensaje de confirmación de que el modelo fue guardado.
# Calcula y muestra la precisión del modelo en el conjunto de entrenamiento.
print(f"📊 Precisión en entrenamiento: {knn.score(X_train, y_train):.2f}")
# Calcula y muestra la precisión del modelo en el conjunto de prueba.
print(f"📊 Precisión en prueba: {knn.score(X_test, y_test):.2f}")