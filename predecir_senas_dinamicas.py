# File: predecir_senas_dinamicas.py
# === predecir_senas_dinamicas.py ===
# Este script predice en tiempo real gestos (palabras o frases) a partir de secuencias de landmarks
# de la mano usando un modelo LSTM previamente entrenado.
# Utiliza la cámara para capturar video, MediaPipe para detectar los landmarks y un modelo de Keras para la predicción.

import cv2              # Importa la librería OpenCV para manejar la cámara y mostrar imágenes.
import numpy as np      # Importa numpy para manejo eficiente de arrays numéricos.
import mediapipe as mp  # Importa la librería de Google MediaPipe para detección de landmarks.
import pickle           # Importa pickle para cargar las etiquetas (nombres de los gestos).
from tensorflow.keras.models import load_model # Importa la función para cargar el modelo entrenado de Keras.

# === CONFIGURACIÓN ===
SECUENCIA_FRAMES = 30   # Define el número de frames que componen una secuencia de gesto. Debe coincidir con el entrenamiento.
MODEL_PATH = "modelo/lstm_senas_dinamicas.h5" # Ruta al archivo del modelo LSTM entrenado.
LABELS_PATH = "modelo/senas_dinamicas_labels.pkl" # Ruta al archivo pickle con las etiquetas (nombres de los gestos).

# === CARGAR MODELO Y ETIQUETAS ===
try:
    model = load_model(MODEL_PATH) # Carga el modelo LSTM desde el archivo especificado.
    print(f"✅ Modelo cargado desde: {MODEL_PATH}") # Mensaje de confirmación.
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}") # Mensaje de error si falla la carga.
    exit() # Sale del script si no se puede cargar el modelo.

try:
    with open(LABELS_PATH, "rb") as f: # Abre el archivo de etiquetas en modo lectura binaria ('rb').
        labels = pickle.load(f) # Carga las etiquetas (lista de nombres de gestos) desde el archivo.
    print(f"✅ Etiquetas cargadas desde: {LABELS_PATH}") # Mensaje de confirmación.
except Exception as e:
    print(f"❌ Error al cargar las etiquetas: {e}") # Mensaje de error si falla la carga.
    exit() # Sale del script si no se pueden cargar las etiquetas.


# === INICIAR MEDIAPIPE ===
mp_hands = mp.solutions.hands # Accede a la solución de detección de manos dentro de MediaPipe.
hands = mp_hands.Hands(
    static_image_mode=False,       # Configura el modo a False para procesamiento de video en tiempo real.
    max_num_hands=2,               # Permite detectar hasta 2 manos en el frame.
    min_detection_confidence=0.5,  # Establece la confianza mínima (50%) para considerar una detección de mano válida.
    min_tracking_confidence=0.5    # Establece la confianza mínima (50%) para seguir una mano ya detectada.
)
mp_draw = mp.solutions.drawing_utils # Accede a las utilidades de dibujo de MediaPipe para visualizar landmarks.

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0) # Inicializa la captura de video desde la cámara predeterminada (índice 0).
if not cap.isOpened(): # Verifica si la cámara se abrió correctamente.
    print("❌ Error: No se pudo abrir la cámara.") # Mensaje de error si la cámara no está disponible.
    exit() # Sale del script si la cámara no se abre.

print("🎥 Mostrá un gesto completo. Presioná 'q' para salir.") # Instrucciones para el usuario.

secuencia = [] # Inicializa una lista para almacenar la secuencia de vectores de landmarks de los últimos frames.

# === BUCLE PRINCIPAL DE PREDICCIÓN ===
while True: # Bucle infinito para procesar los frames de la cámara en tiempo real.
    ret, frame = cap.read() # Lee un frame de la cámara. 'ret' es True si la lectura fue exitosa, 'frame' es la imagen.
    if not ret: # Si no se pudo leer el frame (ej: cámara desconectada), sale del bucle.
        break

    frame = cv2.flip(frame, 1) # Voltea el frame horizontalmente (efecto espejo) para una visualización más intuitiva.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convierte el frame de BGR (formato de OpenCV) a RGB (formato esperado por MediaPipe).
    result = hands.process(rgb) # Procesa el frame con el modelo de detección de manos de MediaPipe.

    # === PROCESAR LANDMARKS Y PREDECIR ===
    if result.multi_hand_landmarks: # Verifica si MediaPipe detectó alguna mano en el frame.
        for hand_landmarks in result.multi_hand_landmarks: # Itera sobre cada mano detectada.
            # Dibuja los landmarks y las conexiones entre ellos en el frame original.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            vector = [] # Inicializa una lista temporal para almacenar las coordenadas de los landmarks de la mano actual.
            for lm in hand_landmarks.landmark: # Itera sobre cada landmark individual (hay 21 por mano).
                vector.extend([lm.x, lm.y, lm.z]) # Añade las coordenadas x, y, z del landmark a la lista temporal.
            vector = np.array(vector) # Convierte la lista temporal a un array numpy.

            secuencia.append(vector) # Añade el vector de landmarks del frame actual a la lista de secuencia.
            # Mantiene la lista de secuencia con un tamaño fijo (SECUENCIA_FRAMES) eliminando el frame más antiguo.
            if len(secuencia) > SECUENCIA_FRAMES:
                secuencia.pop(0)

            # Si la secuencia tiene el número requerido de frames, realiza la predicción.
            if len(secuencia) == SECUENCIA_FRAMES:
                # Prepara la secuencia para la entrada del modelo: añade una dimensión extra al principio (batch size).
                input_seq = np.expand_dims(np.array(secuencia), axis=0)
                pred = model.predict(input_seq)[0] # Realiza la predicción usando el modelo. [0] para obtener el resultado del primer (y único) elemento del batch.
                pred_label_index = np.argmax(pred) # Obtiene el índice de la clase con la mayor probabilidad.
                pred_label = labels[pred_label_index] # Obtiene el nombre del gesto correspondiente al índice predicho.

                # Muestra el gesto predicho en la parte superior del frame.
                cv2.putText(frame, f"Gesto: {pred_label.upper()}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2) # Texto, posición, fuente, escala, color (verde), grosor.
    else:
        # Si no se detecta ninguna mano en el frame, reinicia la secuencia.
        # Esto evita que el modelo intente predecir con secuencias incompletas o sin mano.
        secuencia = []

    # Mostrar leyenda para salir
    cv2.putText(frame, "Presiona 'q' para salir", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2) # Texto, posición, fuente, escala, color (gris), grosor.

    cv2.imshow("Reconocimiento de gestos", frame) # Muestra el frame actual en una ventana llamada "Reconocimiento de gestos".

    # === SALIR DEL BUCLE ===
    # Espera 1 milisegundo por una pulsación de tecla. Si la tecla es 'q', sale del bucle.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🚪 Saliendo por solicitud del usuario.") # Mensaje de salida.
        break # Sale del bucle principal.

# === LIMPIEZA ===
print("✅ Finalizando...") # Mensaje de finalización.
cap.release() # Libera el recurso de la cámara.
cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV.
hands.close() # Cierra el modelo de detección de manos de MediaPipe.
print("👋 Hasta luego.") # Mensaje de despedida.