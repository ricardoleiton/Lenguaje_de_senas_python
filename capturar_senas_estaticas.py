# === capturar_senas_estaticas.py ===
# Este script usa la cámara para capturar los puntos clave (landmarks) de una mano,
# y los guarda como vectores .npy para entrenar un modelo de reconocimiento de señas.
# Permite capturar letras (a-z) y números (0-9). Incluye opción de cancelar con "q".

import cv2              # Para manejar la cámara y mostrar imágenes
import os               # Para manejar carpetas y archivos
import numpy as np      # Para trabajar con vectores
import mediapipe as mp  # Librería de Google para detectar landmarks de manos

# === CONFIGURACIÓN DE MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # Modo video (procesamiento en tiempo real)
    max_num_hands=1,               # Solo detectamos una mano
    min_detection_confidence=0.7,  # Confianza mínima para detectar la mano
    min_tracking_confidence=0.7    # Confianza mínima para seguir la mano
)

# === VALIDACIÓN DE ENTRADA ===
def es_valido(caracter):
    """
    Verifica que la entrada sea una sola letra (a-z) o número (0-9).
    """
    return len(caracter) == 1 and (caracter.isalpha() or caracter.isdigit())

# === PEDIR LETRA O NÚMERO AL USUARIO ===
while True:
    etiqueta = input("▶ Ingresá la letra o número que vas a capturar (a-z, 0-9): ").lower()
    if es_valido(etiqueta):
        break
    print("❌ Entrada inválida. Ingresá una sola letra (a-z) o número (0-9).")

# === CREAR CARPETA PARA GUARDAR VECTORES ===
folder = os.path.join("senas_estaticas", etiqueta)
os.makedirs(folder, exist_ok=True)

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)  # 0: cámara integrada
count = 0                  # Contador de muestras capturadas
total = 100                # Total de muestras por clase

print("📸 Posicioná la mano y presioná 'c' para comenzar, 'q' para cancelar...")

# === FASE DE PREVISUALIZACIÓN ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejar la imagen
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Si se detecta una mano, dibujar los landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Mostrar instrucciones
    cv2.putText(frame, "Presiona 'c' para comenzar o 'q' para salir", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Previsualización", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        break
    elif key == ord('q'):
        print("🚫 Captura cancelada por el usuario.")
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        exit()

cv2.destroyWindow("Previsualización")
print("✅ Comenzando captura de vectores...")

# === FASE DE CAPTURA DE VECTORES ===
while cap.isOpened() and count < total:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])  # Coordenadas x, y, z

            if len(vector) == 63:
                npy_path = os.path.join(folder, f"{etiqueta}_{count}.npy")
                np.save(npy_path, np.array(vector))
                print(f"[✔] Landmark guardado: {npy_path}")
                count += 1

    # Mostrar progreso de la captura
    cv2.putText(frame, f"{etiqueta.upper()} - Muestras: {count}/{total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Capturando landmarks", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("🚫 Captura interrumpida por el usuario.")
        break

# === CIERRE DE RECURSOS ===
print("✅ Captura finalizada.")
cap.release()
cv2.destroyAllWindows()
hands.close()
print("👋 Hasta luego.")