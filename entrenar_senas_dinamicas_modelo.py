# File: entrenar_senas_dinamicas_modelo.py
# === entrenar_senas_dinamicas_modelo.py ===
# Este script carga las secuencias de landmarks de señas dinámicas previamente capturadas,
# prepara los datos y entrena un modelo de red neuronal recurrente (LSTM) para reconocerlas.
# Finalmente, guarda el modelo entrenado y las etiquetas correspondientes.

import os           # Importa la librería os para interactuar con el sistema operativo (listar directorios).
import numpy as np  # Importa numpy para manejo eficiente de arrays numéricos.
import pickle       # Importa pickle para serializar y deserializar objetos Python (guardar las etiquetas).
# Importa funciones y capas necesarias de scikit-learn y TensorFlow/Keras.
from sklearn.model_selection import train_test_split # Para dividir los datos en conjuntos de entrenamiento y prueba.
from tensorflow.keras.models import Sequential       # Para crear un modelo secuencial (capa por capa).
from tensorflow.keras.layers import LSTM, Dense      # Capas LSTM (para secuencias) y Dense (capas completamente conectadas).
from tensorflow.keras.utils import to_categorical    # Para convertir las etiquetas a formato one-hot encoding.

# === CONFIGURACIÓN Y CARGA DE DATOS ===
DATA_DIR = "senas_dinamicas" # Directorio donde se encuentran las secuencias de datos (.npy).
GESTOS = sorted(os.listdir(DATA_DIR)) # Lista de nombres de los gestos (nombres de las subcarpetas en DATA_DIR).
SECUENCIA_FRAMES = 30 # Número de frames por secuencia, debe coincidir con la captura.

X, y = [], [] # Inicializa listas vacías para almacenar los datos (X) y las etiquetas (y).

print("📦 Cargando secuencias...") # Mensaje informativo.

# Itera sobre cada gesto (carpeta) y cada archivo .npy dentro.
for label, gesto in enumerate(GESTOS): # 'label' será un índice numérico (0, 1, 2...) para cada gesto.
    folder = os.path.join(DATA_DIR, gesto) # Construye la ruta completa a la carpeta del gesto.
    for archivo in os.listdir(folder): # Itera sobre los archivos dentro de la carpeta del gesto.
        if archivo.endswith(".npy"): # Verifica si el archivo termina con .npy.
            datos = np.load(os.path.join(folder, archivo)) # Carga los datos del archivo .npy.
            # Verifica que la forma de los datos cargados sea la esperada (SECUENCIA_FRAMES, 63 landmarks).
            if datos.shape == (SECUENCIA_FRAMES, 63):
                X.append(datos) # Añade la secuencia de datos a la lista X.
                y.append(label) # Añade la etiqueta numérica correspondiente a la lista y.

X = np.array(X) # Convierte la lista de datos X a un array numpy.
y = to_categorical(y) # Convierte las etiquetas numéricas y a formato one-hot encoding (ej: 0 -> [1,0,0], 1 -> [0,1,0]).

# === DIVIDIR DATOS ===
# Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%).
# random_state=42 asegura que la división sea la misma cada vez que se ejecuta el script.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODELO LSTM ===
# Define la arquitectura del modelo usando Keras Sequential API.
model = Sequential([
    # Primera capa LSTM: procesa secuencias. return_sequences=True para pasar la secuencia completa a la siguiente capa LSTM.
    # input_shape define la forma de la entrada: (número de frames, número de características por frame).
    LSTM(64, return_sequences=True, input_shape=(SECUENCIA_FRAMES, 63)),
    # Segunda capa LSTM: procesa la secuencia de la capa anterior y devuelve solo el último output (por defecto return_sequences=False).
    LSTM(64),
    # Capa Dense (completamente conectada) con activación ReLU.
    Dense(64, activation='relu'),
    # Capa de salida Dense: tiene tantas neuronas como clases (gestos).
    # La activación 'softmax' produce una distribución de probabilidad sobre las clases.
    Dense(len(GESTOS), activation='softmax')
])

# === COMPILAR MODELO ===
# Configura el proceso de entrenamiento del modelo.
model.compile(optimizer='adam', # Optimizador Adam: un algoritmo de optimización eficiente.
              loss='categorical_crossentropy', # Función de pérdida para clasificación multiclase con one-hot encoding.
              metrics=['accuracy']) # Métrica para evaluar el rendimiento durante el entrenamiento.

# === ENTRENAR MODELO ===
# Entrena el modelo con los datos de entrenamiento.
# epochs=30: El modelo verá todo el conjunto de entrenamiento 30 veces.
# validation_data: Datos para evaluar el modelo al final de cada epoch (no se usan para entrenar).
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# === GUARDAR MODELO Y ETIQUETAS ===
# Guarda el modelo entrenado en formato HDF5.
model.save("modelo/lstm_senas_dinamicas.h5")
# Guarda la lista de nombres de gestos usando pickle. Esto es necesario para saber a qué gesto corresponde cada salida del modelo.
with open("modelo/senas_dinamicas_labels.pkl", "wb") as f: # Abre un archivo en modo escritura binaria ('wb').
    pickle.dump(GESTOS, f) # Serializa la lista GESTOS y la guarda en el archivo.

print("✅ Modelo LSTM guardado.") # Mensaje de confirmación.