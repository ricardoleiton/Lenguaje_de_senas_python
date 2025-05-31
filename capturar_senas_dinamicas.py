# === capturar_senas_dinamicas.py ===
# Este script permite capturar secuencias de video para entrenar un modelo de gestos dinámicos.
# Utiliza MediaPipe para detectar landmarks de la mano y guarda las secuencias como vectores .npy.

import cv2
import os
import numpy as np
import mediapipe as mp

# === CONFIGURACIÓN GENERAL ===
SECUENCIA_FRAMES = 30
CANTIDAD_SECUENCIAS = 20
DATA_DIR = "senas_dinamicas"

# === INICIAR DETECTOR DE MANOS ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

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

    # === CAPTURA AUTOMÁTICA ===
    if captura_activada and contador_sec < CANTIDAD_SECUENCIAS:
        secuencia = []
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
            cv2.imshow("Capturando...", frame_temp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🚫 Captura interrumpida.")
                captura_activada = False
                break

        if len(secuencia) == SECUENCIA_FRAMES:
            np.save(os.path.join(folder, f"{gesto}_{sample_count}.npy"), np.array(secuencia))
            print(f"[✔] Secuencia guardada: {gesto}_{sample_count}.npy")
            sample_count += 1
            contador_sec += 1
        else:
            print("❌ Secuencia incompleta.")

        if contador_sec >= CANTIDAD_SECUENCIAS:
            print("✅ Captura completa.")
            captura_activada = False

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