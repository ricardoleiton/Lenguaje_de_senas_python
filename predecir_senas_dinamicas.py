# === predecir_senas_dinamicas.py ===
# Predice gestos dinámicos en tiempo real y muestra un GIF 3D de ejemplo en otra ventana.

import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

# === CONFIGURACIÓN ===
SECUENCIA_FRAMES = 30
MODEL_PATH = "modelo/lstm_senas_dinamicas.h5"
LABELS_PATH = "modelo/senas_dinamicas_labels.pkl"

# === CARGAR MODELO Y ETIQUETAS ===
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Modelo cargado desde: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

try:
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)
    print(f"✅ Etiquetas cargadas desde: {LABELS_PATH}")
except Exception as e:
    print(f"❌ Error al cargar las etiquetas: {e}")
    exit()

# === FUNCIÓN PARA CARGAR GIF ===
def cargar_gif(path):
    try:
        gif = Image.open(path)
        frames = []
        try:
            while True:
                frames.append(gif.copy().convert("RGB"))
                gif.seek(len(frames))
        except EOFError:
            pass
        return frames
    except Exception as e:
        print(f"❌ No se pudo cargar el GIF: {e}")
        return []

# === INICIAR MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("🎥 Mostrá un gesto. Presioná 'q' para salir.")

# === VARIABLES DE ESTADO ===
secuencia = []
gif_actual = []
gif_actual_label = ""
gif_index = 0

# === BUCLE PRINCIPAL ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            vector = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            vector = np.array(vector)

            secuencia.append(vector)
            if len(secuencia) > SECUENCIA_FRAMES:
                secuencia.pop(0)

            if len(secuencia) == SECUENCIA_FRAMES:
                input_seq = np.expand_dims(np.array(secuencia), axis=0)
                pred = model.predict(input_seq)[0]
                pred_label_index = np.argmax(pred)
                pred_label = labels[pred_label_index]

                cv2.putText(frame, f"Gesto: {pred_label.upper()}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                # Cargar GIF si cambió el gesto
                gif_path = f"gifs/{pred_label}.gif"
                if pred_label != gif_actual_label:
                    gif_actual = cargar_gif(gif_path)
                    gif_actual_label = pred_label
                    gif_index = 0
    else:
        secuencia = []

    # Mostrar ventana con la cámara y la predicción
    cv2.imshow("Reconocimiento de gestos", frame)

    # Mostrar ventana separada con el GIF
    if gif_actual:
        gif_frame = np.array(gif_actual[gif_index % len(gif_actual)])
        gif_index += 1
        gif_frame = cv2.cvtColor(gif_frame, cv2.COLOR_RGB2BGR)  # Pillow usa RGB, OpenCV BGR
        gif_frame = cv2.resize(gif_frame, (300, 300))
        cv2.imshow("Ejemplo 3D", gif_frame)

    # Tecla para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🚪 Saliendo por solicitud del usuario.")
        break

# === CIERRE DE RECURSOS ===
cap.release()
cv2.destroyAllWindows()
hands.close()
print("👋 Hasta luego.")