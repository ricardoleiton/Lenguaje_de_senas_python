# === predecir_todas_letras.py ===
# Este script usa la cámara para predecir en tiempo real la letra que se está mostrando
# con la mano, usando el modelo entrenado previamente.

import cv2
import numpy as np
import mediapipe as mp
import pickle

# === CONFIGURACIÓN ===
MODEL_PATH = "modelo/knn_landmark.pkl"
letters = "abcdefghijklmnopqrstuvwxyz"

# === CARGAR MODELO ENTRENADO ===
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# === CONFIGURAR MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# === ACTIVAR CÁMARA ===
cap = cv2.VideoCapture(1)
print("🎥 Cámara activa. Mostrá una seña. Presioná 'q' para salir.")

# === PROCESAR VIDEO EN VIVO ===
while cap.isOpened():
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

            if len(vector) == 63:
                vector = np.array(vector).reshape(1, -1)
                prediction = model.predict(vector)[0]
                predicted_letter = letters[prediction]

                # Mostrar letra en pantalla
                cv2.putText(frame, f"Letra: {predicted_letter.upper()}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento de señas (a-z)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CERRAR TODO ===
cap.release()
cv2.destroyAllWindows()
hands.close()