# File: predecir_senas_estaticas.py
# === predecir_senas_estaticas.py ===
# Este script usa la cámara para predecir en tiempo real la letra que se está mostrando
# con la mano, usando el modelo entrenado previamente.
# Utiliza MediaPipe para detectar los landmarks de la mano y un modelo KNN para la predicción.

import cv2              # Importa la librería OpenCV para manejar la cámara y mostrar imágenes.
import numpy as np      # Importa numpy para manejo eficiente de arrays numéricos.
import mediapipe as mp  # Importa la librería de Google MediaPipe para detección de landmarks.
import pickle           # Importa pickle para cargar el modelo entrenado.

# === CONFIGURACIÓN ===
MODEL_PATH = "modelo/knn_senas_estaticas.pkl" # Ruta al archivo del modelo KNN entrenado.
letters = "abcdefghijklmnopqrstuvwxyz01236789" # Cadena con las letras y números que el modelo puede predecir.

# === CARGAR MODELO ENTRENADO ===
try:
    with open(MODEL_PATH, "rb") as f: # Abre el archivo del modelo en modo lectura binaria ('rb').
        model = pickle.load(f) # Carga el modelo entrenado desde el archivo.
    print(f"✅ Modelo cargado desde: {MODEL_PATH}") # Mensaje de confirmación.
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo del modelo en {MODEL_PATH}") # Mensaje de error si el archivo no existe.
    print("Asegúrate de haber entrenado el modelo primero ejecutando 'entrenar_senas_estaticas_modelo.py'.")
    exit() # Sale del script si el modelo no se encuentra.
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}") # Mensaje de error para otras excepciones.
    exit() # Sale del script si hay otro error al cargar el modelo.


# === CONFIGURAR MEDIAPIPE ===
mp_hands = mp.solutions.hands # Accede a la solución de detección de manos dentro de MediaPipe.
hands = mp_hands.Hands(
    static_image_mode=False,       # Configura el modo a False para procesamiento de video en tiempo real.
    max_num_hands=1,               # Limita la detección a una sola mano en el frame.
    min_detection_confidence=0.5,  # Establece la confianza mínima (50%) para considerar una detección de mano válida.
    min_tracking_confidence=0.5    # Establece la confianza mínima (50%) para seguir una mano ya detectada.
)
mp_draw = mp.solutions.drawing_utils # Accede a las utilidades de dibujo de MediaPipe para visualizar landmarks.

# === ACTIVAR CÁMARA ===
cap = cv2.VideoCapture(0) # Inicializa la captura de video desde la cámara predeterminada (índice 0).
if not cap.isOpened(): # Verifica si la cámara se abrió correctamente.
    print("❌ Error: No se pudo abrir la cámara.") # Mensaje de error si la cámara no está disponible.
    exit() # Sale del script si la cámara no se abre.

print("🎥 Cámara activa. Mostrá una seña. Presioná 'q' para salir.") # Instrucciones para el usuario.

# === PROCESAR VIDEO EN VIVO ===
while cap.isOpened(): # Bucle principal: continúa mientras la cámara esté abierta.
    ret, frame = cap.read() # Lee un frame de la cámara. 'ret' es True si la lectura fue exitosa, 'frame' es la imagen.
    if not ret: # Si no se pudo leer el frame (ej: cámara desconectada), sale del bucle.
        break

    frame = cv2.flip(frame, 1) # Voltea el frame horizontalmente (efecto espejo) para una visualización más intuitiva.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convierte el frame de BGR (formato de OpenCV) a RGB (formato esperado por MediaPipe).
    result = hands.process(rgb) # Procesa el frame con el modelo de detección de manos de MediaPipe.

    # === PROCESAR LANDMARKS Y PREDECIR ===
    if result.multi_hand_landmarks: # Verifica si MediaPipe detectó alguna mano en el frame.
        for hand_landmarks in result.multi_hand_landmarks: # Itera sobre cada mano detectada (aunque configuramos para 1).
            # Dibuja los landmarks y las conexiones entre ellos en el frame original.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            vector = [] # Inicializa una lista vacía para almacenar las coordenadas de los landmarks.
            for lm in hand_landmarks.landmark: # Itera sobre cada landmark individual (hay 21 por mano).
                # Añade las coordenadas x, y, z del landmark al vector.
                # Nota: Para señas estáticas, a veces solo se usan x, y, o se normalizan. Aquí se usan x, y, z.
                vector.extend([lm.x, lm.y, lm.z])

            # Verifica que el vector tenga 63 elementos (21 landmarks * 3 coordenadas).
            if len(vector) == 63:
                vector = np.array(vector).reshape(1, -1) # Convierte la lista a un array numpy y lo remodela para que tenga la forma (1, 63), adecuada para la entrada del modelo.
                prediction = model.predict(vector)[0] # Realiza la predicción usando el modelo KNN. [0] para obtener el resultado del primer (y único) elemento del array de predicciones.
                predicted_letter = letters[prediction] # Obtiene la letra o número correspondiente al índice predicho usando la cadena 'letters'.

                # Mostrar letra predicha en pantalla
                cv2.putText(frame, f"Letra: {predicted_letter.upper()}", (10, 40), # Texto a mostrar y su posición.
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2) # Fuente, escala, color (verde) y grosor del texto.

    # Mostrar leyenda para salir
    cv2.putText(frame, "Presiona 'q' para salir", (10, 80), # Texto a mostrar y su posición.
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2) # Fuente, escala, color (gris) y grosor del texto.

    cv2.imshow("Reconocimiento de señas (a-z, 0-9)", frame) # Muestra el frame actual en una ventana llamada "Reconocimiento de señas".

    # === SALIR DEL BUCLE ===
    # Espera 1 milisegundo por una pulsación de tecla. Si la tecla es 'q', sale del bucle.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🚪 Saliendo por solicitud del usuario.") # Mensaje de salida.
        break # Sale del bucle principal.

# === CIERRE DE RECURSOS ===
print("✅ Finalizando...") # Mensaje de finalización.
cap.release() # Libera el recurso de la cámara.
cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV.
hands.close() # Cierra el modelo de detección de manos de MediaPipe.
print("👋 Hasta luego.") # Mensaje de despedida.