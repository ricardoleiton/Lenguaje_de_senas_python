# File: capturar_senas_dinamicas.py
# === capturar_senas_dinamicas.py ===
# Este script permite capturar secuencias de video para entrenar un modelo de gestos dinámicos.
# Utiliza MediaPipe para detectar landmarks de la mano y guarda las secuencias como vectores .npy.
# ADEMÁS, genera automáticamente un GIF usando los frames reales captados por la cámara (para uso educativo con profesionales de lenguaje de señas).

import cv2  # Importa la librería OpenCV para manejo de video y procesamiento de imágenes.
import os  # Importa la librería os para interactuar con el sistema operativo (crear carpetas, etc.).
import numpy as np  # Importa la librería numpy para manejo eficiente de arrays numéricos.
import mediapipe as mp  # Importa la librería MediaPipe para detección de landmarks.
from PIL import Image  # Para manipular imágenes y guardar GIFs.

# === CONFIGURACIÓN GENERAL ===
SECUENCIA_FRAMES = 30  # Define el número de frames por cada secuencia de gesto a capturar.
CANTIDAD_SECUENCIAS = 20  # Define la cantidad total de secuencias a capturar por cada gesto.
DATA_DIR = "senas_dinamicas"  # Define el nombre del directorio donde se guardarán los datos capturados.

# === INICIAR DETECTOR DE MANOS ===
mp_hands = mp.solutions.hands  # Accede a la solución de detección de manos de MediaPipe.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)  # Inicializa el modelo de detección de manos.
mp_draw = mp.solutions.drawing_utils  # Accede a las utilidades de dibujo de MediaPipe para visualizar landmarks.

# === FUNCIÓN PARA GENERAR GIF DESDE LOS FRAMES REALES ===
def generar_gif_desde_frames(frames, ruta_gif):
    """
    Genera un GIF animado a partir de los frames reales capturados por la cámara.
    """
    gif_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((300, 300)) for f in frames]
    os.makedirs("gifs", exist_ok=True)
    gif_frames[0].save(ruta_gif, save_all=True, append_images=gif_frames[1:], duration=50, loop=0)
    print(f"🎞️ GIF generado desde cámara: {ruta_gif}")

# === INGRESAR NOMBRE DEL GESTO ===
while True:
    gesto = input("▶ Ingresá la palabra o frase a capturar (sin espacios): ").lower()
    if gesto.isalpha():
        break
    print("❌ Entrada inválida. Solo letras, sin espacios ni números.")

# === CREAR CARPETA PARA EL GESTO ===
folder = os.path.join(DATA_DIR, gesto)
os.makedirs(folder, exist_ok=True)
sample_count = len(os.listdir(folder))

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
print("📸 Posicioná la mano. Presioná 'c' para comenzar o 'q' para salir.")

captura_activada = False
contador_sec = 0

# === BUCLE PRINCIPAL ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    mensaje = f"Gesto: {gesto} - Secuencias guardadas: {contador_sec}/{CANTIDAD_SECUENCIAS}"
    instrucciones = "Presiona 'c' para capturar | 'q' para salir"
    cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, instrucciones, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    if captura_activada and contador_sec < CANTIDAD_SECUENCIAS:
        secuencia = []  # Lista para almacenar vectores de landmarks
        frames_para_gif = []  # Lista para almacenar los frames reales para el GIF

        for _ in range(SECUENCIA_FRAMES):
            ret, frame_temp = cap.read()
            if not ret:
                break

            frame_temp = cv2.flip(frame_temp, 1)
            rgb = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            vector = np.zeros(63)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    temp = []
                    for lm in hand_landmarks.landmark:
                        temp.extend([lm.x, lm.y, lm.z])
                    vector = np.array(temp)

            secuencia.append(vector)
            frames_para_gif.append(frame_temp.copy())

            # Mostrar el frame de captura en la misma ventana principal
            mensaje = f"Gesto: {gesto} - Capturando {len(secuencia)}/{SECUENCIA_FRAMES}"
            cv2.putText(frame_temp, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Vista previa", frame_temp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🚫 Captura interrumpida.")
                captura_activada = False
                break

        if len(secuencia) == SECUENCIA_FRAMES:
            nombre_archivo = f"{gesto}_{sample_count}.npy"
            np.save(os.path.join(folder, nombre_archivo), np.array(secuencia))
            print(f"[✔] Secuencia guardada: {nombre_archivo}")
            sample_count += 1
            contador_sec += 1

            # === GENERAR GIF DESDE LOS FRAMES REALES ===
            if contador_sec == 1:
                ruta_gif = os.path.join("gifs", f"{gesto}.gif")
                generar_gif_desde_frames(frames_para_gif, ruta_gif)
        else:
            print("❌ Secuencia incompleta.")

        if contador_sec >= CANTIDAD_SECUENCIAS:
            print("✅ Captura completa.")
            captura_activada = False
    else:
        cv2.imshow("Vista previa", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        captura_activada = True
        print("⏳ Iniciando captura automática...")
    elif key == ord('q'):
        print("🚪 Saliendo por solicitud del usuario.")
        break

# === LIBERAR RECURSOS ===
cap.release()
cv2.destroyAllWindows()
hands.close()
print("👋 Finalizado.")