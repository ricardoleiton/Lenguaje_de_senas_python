# === capturar_landmarks.py ===
# Este script usa la cámara para capturar los puntos clave (landmarks) de una mano,
# los guarda como vectores .npy para entrenar un modelo de reconocimiento de señas.

import cv2  # Para manejar la cámara y mostrar imágenes
import os   # Para manejar carpetas y archivos
import numpy as np  # Para trabajar con vectores
import mediapipe as mp  # Librería de Google para detectar landmarks de manos

# === CONFIGURACIÓN DE MEDIAPIPE ===
mp_hands = mp.solutions.hands # Le decimos a mediapipe que vamos a usar la detección de manos
hands = mp_hands.Hands( # Inicializamos la detección de manos con los siguientes parámetros
    static_image_mode=False,  # Modo video en tiempo real
    max_num_hands=1,          # Solo se procesa una mano
    min_detection_confidence=0.7, # Confianza mínima para detección de manos 70%
    min_tracking_confidence=0.7 # Confianza mínima para seguimiento de manos 70%
)

# === PEDIR LETRA A CAPTURAR ===
while True: # Pedimos al usuario que ingrese una letra
    current_letter = input("▶ Ingresá la letra que vas a capturar (a-z): ").lower() # Convertimos la letra a minúscula
    if len(current_letter) == 1 and current_letter.isalpha(): # Verificamos que sea una sola letra
        break # Si es válida, salimos del bucle
    print("❌ Letra inválida. Ingresá una sola letra (a-z).") # Si no es válida, mostramos un mensaje de error

# === CREAR CARPETA PARA GUARDAR LANDMARKS ===
folder = os.path.join("landmarks_data", current_letter) # Creamos la carpeta donde se guardarán los landmarks
os.makedirs(folder, exist_ok=True) # Si la carpeta ya existe, no hacemos nada

# === ACTIVAR CÁMARA ===
cap = cv2.VideoCapture(1) # Inicializamos la cámara (0 para la cámara integrada, 1 para una externa usb)
count = 0 # Contador de muestras
total = 100  # Número total de muestras a capturar. 100 vectores por letra

print("📸 Posicioná la mano y presioná 'c' para comenzar la captura...") # Aviso en consola al usuario

# === ESPERAR QUE EL USUARIO PRESIONE 'c' ===
while True: # Esperamos a que el usuario presione 'c' para comenzar la captura
    ret, frame = cap.read() # Leemos un frame de la cámara, Frame es la imagen capturada
    if not ret: # Si no se puede leer el frame, salimos del bucle 
        break # Salimos del bucle

    frame = cv2.flip(frame, 1)  # Volteamos la imagen horizontalmente como en un espejo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertimos la imagen de BGR a RGB para que MediaPipe la entienda
    result = hands.process(rgb) # Procesamos la imagen con MediaPipe para detectar la mano

    # Dibujar landmarks si se detecta una mano
    if result.multi_hand_landmarks: # Si se detecta una mano
        for hand_landmarks in result.multi_hand_landmarks: # Para cada mano detectada
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Dibujamos los landmarks y las conexiones entre ellos

    cv2.putText(frame, "Presiona 'c' para comenzar", (10, 30), # Mostramos un mensaje en la imagen
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2) # Texto amarillo
    cv2.imshow("Previsualización", frame) # Mostramos la imagen en una ventana llamada "Previsualización"
    key = cv2.waitKey(1) & 0xFF # Esperamos 1 ms para que el usuario pueda presionar una tecla
    if key == ord('c'): # Si el usuario presiona 'c'
        break # Salimos del bucle

cv2.destroyWindow("Previsualización") # Cerramos la ventana de previsualización
print("✅ Comenzando captura...") # Aviso en consola al usuario

# === CAPTURAR LANDMARKS Y GUARDAR COMO VECTORES ===
while cap.isOpened() and count < total: # Mientras la cámara esté abierta y no se hayan capturado todas las muestras
    ret, frame = cap.read() # Leemos un frame de la cámara
    if not ret: # Si no se puede leer el frame, salimos del bucle
        break # Salimos del bucle

    frame = cv2.flip(frame, 1) # Volteamos la imagen horizontalmente como en un espejo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertimos la imagen de BGR a RGB para que MediaPipe la entienda
    result = hands.process(rgb) # Procesamos la imagen con MediaPipe para detectar la mano

    if result.multi_hand_landmarks: # Si se detecta una mano
        for hand_landmarks in result.multi_hand_landmarks: # Para cada mano detectada
            vector = [] # Inicializamos un vector vacío para guardar los landmarks
            for lm in hand_landmarks.landmark: # Para cada landmark de la mano. Guardamos las coordenadas x, y, z del landmark en el vector
                vector.extend([lm.x, lm.y, lm.z])  # Guardar x, y, z

            # Guardar como archivo .npy
            npy_path = os.path.join(folder, f"{current_letter}_{count}.npy") # Guardamos el vector como un archivo .npy
            np.save(npy_path, np.array(vector)) # Guardamos el vector como un archivo .npy
            print(f"[✔] Landmark guardado: {npy_path}") # Mensaje en consola al usuario
            count += 1 # Aumentamos el contador de muestras

    cv2.putText(frame, f"{current_letter.upper()} - Muestras: {count}/{total}", (10, 30), # Mostramos un mensaje en la imagen
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Texto verde
    cv2.imshow("Capturando landmarks", frame) # Mostramos la imagen en una ventana llamada "Capturando landmarks". Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): # Si el usuario presiona 'q'
        break # Salimos del bucle

print("✅ Captura finalizada.") # Aviso en consola al usuario
cap.release() # Liberamos la cámara
cv2.destroyAllWindows() # Cerramos todas las ventanas
hands.close() # Cerramos la detección de manos
print("👋 Hasta luego!") # Mensaje de despedida