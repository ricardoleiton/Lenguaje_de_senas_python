# File: capturar_senas_dinamicas.py
# === capturar_senas_dinamicas.py ===
# Este script permite capturar secuencias de video para entrenar un modelo de gestos dinámicos.
# Utiliza MediaPipe para detectar landmarks de la mano y guarda las secuencias como vectores .npy.

import cv2 # Importa la librería OpenCV para manejo de video y procesamiento de imágenes.
import os # Importa la librería os para interactuar con el sistema operativo (crear carpetas, etc.).
import numpy as np # Importa la librería numpy para manejo eficiente de arrays numéricos.
import mediapipe as mp # Importa la librería MediaPipe para detección de landmarks.

# === CONFIGURACIÓN GENERAL ===
SECUENCIA_FRAMES = 30 # Define el número de frames por cada secuencia de gesto a capturar.
CANTIDAD_SECUENCIAS = 20 # Define la cantidad total de secuencias a capturar por cada gesto.
DATA_DIR = "senas_dinamicas" # Define el nombre del directorio donde se guardarán los datos capturados.

# === INICIAR DETECTOR DE MANOS ===
mp_hands = mp.solutions.hands # Accede a la solución de detección de manos de MediaPipe.
# Inicializa el modelo de detección de manos.
# static_image_mode=False: Modo para procesamiento de video (más rápido).
# max_num_hands=2: Detecta hasta 2 manos.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils # Accede a las utilidades de dibujo de MediaPipe para visualizar landmarks.

# === INGRESAR NOMBRE DEL GESTO ===
while True: # Bucle para solicitar y validar el nombre del gesto.
    gesto = input("▶ Ingresá la palabra o frase a capturar (sin espacios): ").lower() # Solicita al usuario el nombre del gesto y lo convierte a minúsculas.
    if gesto.isalpha(): # Verifica si la entrada consiste solo en letras.
        break # Si es válido, sale del bucle.
    print("❌ Entrada inválida. Solo letras, sin espacios ni números.") # Muestra un mensaje de error si la entrada es inválida.

# === CREAR CARPETA PARA EL GESTO ===
folder = os.path.join(DATA_DIR, gesto) # Construye la ruta completa de la carpeta para el gesto.
os.makedirs(folder, exist_ok=True) # Crea la carpeta si no existe.
sample_count = len(os.listdir(folder)) # Cuenta cuántas secuencias ya existen en la carpeta para nombrar las nuevas.

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0) # Inicializa la captura de video desde la cámara predeterminada (índice 0).
print("📸 Posicioná la mano. Presioná 'c' para comenzar o 'q' para salir.") # Instrucciones para el usuario.

captura_activada = False # Bandera para controlar si la captura automática está activa.
contador_sec = 0 # Contador para llevar el registro de las secuencias guardadas en la sesión actual.

# === BUCLE PRINCIPAL ===
while True: # Bucle principal para procesar los frames de la cámara.
    ret, frame = cap.read() # Lee un frame de la cámara. 'ret' es True si la lectura fue exitosa, 'frame' es la imagen.
    if not ret: # Si no se pudo leer el frame, sale del bucle.
        break

    frame = cv2.flip(frame, 1) # Voltea el frame horizontalmente (efecto espejo).
    mensaje = f"Gesto: {gesto} - Secuencias guardadas: {contador_sec}/{CANTIDAD_SECUENCIAS}" # Crea el mensaje de estado.
    instrucciones = "Presiona 'c' para capturar | 'q' para salir" # Crea el mensaje de instrucciones.
    # Muestra el mensaje de estado en el frame.
    cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Muestra el mensaje de instrucciones en el frame.
    cv2.putText(frame, instrucciones, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # === CAPTURA AUTOMÁTICA ===
    # Verifica si la captura está activada y si aún no se ha alcanzado la cantidad deseada de secuencias.
    if captura_activada and contador_sec < CANTIDAD_SECUENCIAS:
        secuencia = [] # Inicializa una lista para almacenar los vectores de landmarks de la secuencia actual.
        for _ in range(SECUENCIA_FRAMES): # Bucle para capturar los frames de una secuencia.
            ret, frame_temp = cap.read() # Lee un frame temporal para la secuencia.
            if not ret: # Si no se pudo leer el frame, sale del bucle de secuencia.
                break

            frame_temp = cv2.flip(frame_temp, 1) # Voltea el frame temporal.
            rgb = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB) # Convierte el frame a formato RGB (necesario para MediaPipe).
            result = hands.process(rgb) # Procesa el frame con el modelo de detección de manos de MediaPipe.

            vector = np.zeros(63) # Inicializa un vector numpy de ceros para almacenar los landmarks (21 landmarks * 3 coordenadas = 63).
            if result.multi_hand_landmarks: # Verifica si se detectaron manos en el frame.
                for hand_landmarks in result.multi_hand_landmarks: # Itera sobre cada mano detectada.
                    temp = [] # Lista temporal para almacenar las coordenadas de los landmarks de una mano.
                    for lm in hand_landmarks.landmark: # Itera sobre cada landmark de la mano.
                        temp.extend([lm.x, lm.y, lm.z]) # Añade las coordenadas x, y, z del landmark a la lista temporal.
                    vector = np.array(temp) # Convierte la lista temporal a un array numpy y lo asigna al vector.

            secuencia.append(vector) # Añade el vector de landmarks del frame actual a la secuencia.
            cv2.imshow("Capturando...", frame_temp) # Muestra el frame temporal en una ventana separada durante la captura.

            # Permite interrumpir la captura de secuencia presionando 'q'.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🚫 Captura interrumpida.") # Mensaje de interrupción.
                captura_activada = False # Desactiva la captura automática.
                break # Sale del bucle de secuencia.

        if len(secuencia) == SECUENCIA_FRAMES: # Verifica si la secuencia capturada tiene el número correcto de frames.
            # Guarda la secuencia como un archivo .npy en la carpeta del gesto.
            np.save(os.path.join(folder, f"{gesto}_{sample_count}.npy"), np.array(secuencia))
            print(f"[✔] Secuencia guardada: {gesto}_{sample_count}.npy") # Mensaje de confirmación de guardado.
            sample_count += 1 # Incrementa el contador de secuencias guardadas para el nombre del archivo.
            contador_sec += 1 # Incrementa el contador de secuencias guardadas en la sesión actual.
        else:
            print("❌ Secuencia incompleta.") # Mensaje si la secuencia no tiene el número esperado de frames.

        if contador_sec >= CANTIDAD_SECUENCIAS: # Verifica si se ha alcanzado la cantidad total de secuencias deseadas.
            print("✅ Captura completa.") # Mensaje de captura completa.
            captura_activada = False # Desactiva la captura automática.

    cv2.imshow("Vista previa", frame) # Muestra el frame actual de la cámara en la ventana principal.
    key = cv2.waitKey(1) & 0xFF # Espera 1 ms por una pulsación de tecla.

    if key == ord('c'): # Si la tecla presionada es 'c'.
        captura_activada = True # Activa la bandera de captura automática.
        print("⏳ Iniciando captura automática...") # Mensaje de inicio de captura.
    elif key == ord('q'): # Si la tecla presionada es 'q'.
        print("🚪 Saliendo por solicitud del usuario.") # Mensaje de salida.
        break # Sale del bucle principal.

# === LIBERAR RECURSOS ===
cap.release() # Libera el recurso de la cámara.
cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV.
hands.close() # Cierra el modelo de detección de manos de MediaPipe.
print("👋 Finalizado.") # Mensaje de finalización del script.