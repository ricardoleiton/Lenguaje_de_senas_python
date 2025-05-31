# === predecir_senas_dinamicas.py ===
# Este script predice en tiempo real gestos (palabras o frases) a partir de secuencias de landmarks usando un modelo LSTM entrenado.

import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model

# === CONFIGURACIÓN ===
SECUENCIA_FRAMES = 30
MODEL_PATH = "modelo/lstm_senas_dinamicas.h5"
LABELS_PATH = "modelo/senas_dinamicas_labels.pkl"

# === CARGAR MODELO Y ETIQUETAS ===
model = load_model(MODEL_PATH)
with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)

# === INICIAR MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
print("🎥 Mostrá un gesto completo. Presioná 'q' para salir.")

secuencia = []

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

            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])
            vector = np.array(vector)

            secuencia.append(vector)
            if len(secuencia) > SECUENCIA_FRAMES:
                secuencia.pop(0)

            if len(secuencia) == SECUENCIA_FRAMES:
                input_seq = np.expand_dims(np.array(secuencia), axis=0)
                pred = model.predict(input_seq)[0]
                pred_label = labels[np.argmax(pred)]

                cv2.putText(frame, f"Gesto: {pred_label.upper()}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        secuencia = []  # Si no se detecta mano, reiniciar secuencia

    # Mostrar leyenda para salir
    cv2.putText(frame, "Presiona 'q' para salir", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Reconocimiento de gestos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === LIMPIEZA ===
cap.release()
cv2.destroyAllWindows()
hands.close()