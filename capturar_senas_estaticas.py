# File: capturar_senas_estaticas.py
# === capturar_senas_estaticas.py ===
# Este script usa la cámara para capturar los puntos clave (landmarks) de una mano,
# y los guarda como vectores .npy para entrenar un modelo de reconocimiento de señas.
# Permite capturar letras (a-z) y números (0-9). Incluye opción de cancelar con "q".

import cv2              # Importa la librería OpenCV para manejar la cámara y mostrar imágenes.
import os               # Importa la librería os para manejar carpetas y archivos del sistema operativo.
import numpy as np      # Importa la librería numpy para trabajar eficientemente con vectores y arrays numéricos.
import mediapipe as mp  # Importa la librería de Google MediaPipe para detectar landmarks de manos.

# === CONFIGURACIÓN DE MEDIAPIPE ===
mp_hands = mp.solutions.hands # Accede a la solución de detección de manos dentro de MediaPipe.
hands = mp_hands.Hands(
    static_image_mode=False,       # Configura el modo a False para procesamiento de video en tiempo real.
    max_num_hands=1,               # Limita la detección a una sola mano en el frame.
    min_detection_confidence=0.7,  # Establece la confianza mínima (70%) para considerar una detección de mano válida.
    min_tracking_confidence=0.7    # Establece la confianza mínima (70%) para seguir una mano ya detectada.
)

# === VALIDACIÓN DE ENTRADA ===
def es_valido(caracter):
    """
    Verifica que la entrada sea una sola letra (a-z) o número (0-9).
    """
    # Comprueba si la longitud de la cadena es 1 Y si es una letra (isalpha) O un dígito (isdigit).
    return len(caracter) == 1 and (caracter.isalpha() or caracter.isdigit())

# === PEDIR LETRA O NÚMERO AL USUARIO ===
while True: # Bucle infinito para solicitar la entrada hasta que sea válida.
    etiqueta = input("▶ Ingresá la letra o número que vas a capturar (a-z, 0-9): ").lower() # Pide al usuario la etiqueta y la convierte a minúsculas.
    if es_valido(etiqueta): # Llama a la función es_valido para verificar la entrada.
        break # Si la entrada es válida, sale del bucle.
    print("❌ Entrada inválida. Ingresá una sola letra (a-z) o número (0-9).") # Muestra un mensaje de error si la entrada no es válida.

# === CREAR CARPETA PARA GUARDAR VECTORES ===
folder = os.path.join("senas_estaticas", etiqueta) # Construye la ruta de la carpeta donde se guardarán los datos (ej: senas_estaticas/a).
os.makedirs(folder, exist_ok=True) # Crea la carpeta. exist_ok=True evita un error si la carpeta ya existe.

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)  # Inicializa la captura de video desde la cámara predeterminada (índice 0).
count = 0                  # Inicializa un contador para el número de muestras capturadas para la etiqueta actual.
total = 100                # Define el número total de muestras que se desean capturar por cada clase (letra/número).

print("📸 Posicioná la mano y presioná 'c' para comenzar, 'q' para cancelar...") # Instrucciones iniciales para el usuario.

# === FASE DE PREVISUALIZACIÓN ===
while True: # Bucle para mostrar la vista previa de la cámara antes de empezar la captura.
    ret, frame = cap.read() # Lee un frame de la cámara. 'ret' es True si la lectura fue exitosa, 'frame' es la imagen.
    if not ret: # Si no se pudo leer el frame (ej: cámara desconectada), sale del bucle.
        break

    frame = cv2.flip(frame, 1)  # Voltea el frame horizontalmente (efecto espejo) para una visualización más intuitiva.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convierte el frame de BGR (formato de OpenCV) a RGB (formato esperado por MediaPipe).
    result = hands.process(rgb) # Procesa el frame con el modelo de detección de manos de MediaPipe.

    # Si se detecta una mano, dibujar los landmarks
    if result.multi_hand_landmarks: # Verifica si MediaPipe detectó alguna mano en el frame.
        for hand_landmarks in result.multi_hand_landmarks: # Itera sobre cada mano detectada (aunque configuramos para 1).
            mp.solutions.drawing_utils.draw_landmarks( # Utiliza las utilidades de dibujo de MediaPipe.
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS # Dibuja los landmarks y las conexiones entre ellos en el frame original.
            )

    # Mostrar instrucciones en la ventana de previsualización.
    cv2.putText(frame, "Presiona 'c' para comenzar o 'q' para salir", (10, 30), # Texto a mostrar y su posición.
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) # Fuente, escala, color (cian) y grosor del texto.
    cv2.imshow("Previsualización", frame) # Muestra el frame actual en una ventana llamada "Previsualización".

    key = cv2.waitKey(1) & 0xFF # Espera 1 milisegundo por una pulsación de tecla y obtiene su código ASCII.
    if key == ord('c'): # Si la tecla presionada es 'c'.
        break # Sale del bucle de previsualización para comenzar la captura.
    elif key == ord('q'): # Si la tecla presionada es 'q'.
        print("🚫 Captura cancelada por el usuario.") # Mensaje de cancelación.
        cap.release() # Libera el recurso de la cámara.
        cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV.
        hands.close() # Cierra el modelo de detección de manos de MediaPipe.
        exit() # Termina la ejecución del script.

cv2.destroyWindow("Previsualización") # Cierra la ventana de previsualización una vez que se sale del bucle.
print("✅ Comenzando captura de vectores...") # Mensaje indicando que la fase de captura ha comenzado.

# === FASE DE CAPTURA DE VECTORES ===
while cap.isOpened() and count < total: # Bucle principal de captura: continúa mientras la cámara esté abierta Y no se haya alcanzado el total de muestras.
    ret, frame = cap.read() # Lee un nuevo frame de la cámara.
    if not ret: # Si no se pudo leer el frame, sale del bucle.
        break

    frame = cv2.flip(frame, 1) # Voltea el frame horizontalmente.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convierte el frame a RGB.
    result = hands.process(rgb) # Procesa el frame con MediaPipe.

    if result.multi_hand_landmarks: # Si se detectó una mano.
        for hand_landmarks in result.multi_hand_landmarks: # Itera sobre los landmarks de la mano detectada.
            vector = [] # Inicializa una lista vacía para almacenar las coordenadas de los landmarks.
            for lm in hand_landmarks.landmark: # Itera sobre cada landmark individual (hay 21 por mano).
                vector.extend([lm.x, lm.y, lm.z])  # Añade las coordenadas x, y, z del landmark al vector.

            if len(vector) == 63: # Verifica que el vector tenga 63 elementos (21 landmarks * 3 coordenadas).
                npy_path = os.path.join(folder, f"{etiqueta}_{count}.npy") # Construye la ruta completa para guardar el archivo .npy (ej: senas_estaticas/a/a_0.npy).
                np.save(npy_path, np.array(vector)) # Guarda el vector como un archivo numpy (.npy) en la ruta especificada.
                print(f"[✔] Landmark guardado: {npy_path}") # Mensaje de confirmación de guardado.
                count += 1 # Incrementa el contador de muestras guardadas.

    # Mostrar progreso de la captura en la ventana de captura.
    cv2.putText(frame, f"{etiqueta.upper()} - Muestras: {count}/{total}", (10, 30), # Texto de progreso.
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Fuente, escala, color (verde) y grosor.
    cv2.imshow("Capturando landmarks", frame) # Muestra el frame actual en la ventana de captura.

    key = cv2.waitKey(1) & 0xFF # Espera 1 ms por una pulsación de tecla.
    if key == ord('q'): # Si la tecla presionada es 'q'.
        print("🚫 Captura interrumpida por el usuario.") # Mensaje de interrupción.
        break # Sale del bucle de captura.

# === CIERRE DE RECURSOS ===
print("✅ Captura finalizada.") # Mensaje indicando que la captura ha terminado (ya sea por completar o interrumpir).
cap.release() # Libera el recurso de la cámara.
cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV.
hands.close() # Cierra el modelo de detección de manos de MediaPipe.
print("👋 Hasta luego.") # Mensaje de despedida.