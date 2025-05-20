
# 🤟 Lenguaje de Señas con Python

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow)

Este proyecto permite capturar, entrenar y predecir señas usando Python y MediaPipe.  
Ideal para aplicaciones educativas o inclusivas en el reconocimiento del lenguaje de señas.

---

## 📁 Estructura del Proyecto

- 📸 `capturar_landmarks.py`: Captura los puntos de referencia (landmarks) de la mano con MediaPipe.
- 🧠 `entrenar_modelo.py`: Entrena un modelo de aprendizaje automático con los datos capturados.
- 🔍 `predecir_todas_letras.py`: Predice en tiempo real la letra mostrada en lenguaje de señas.

---

## 🚀 Tecnologías Utilizadas

- Python 🐍
- OpenCV 🎥
- MediaPipe ✋
- NumPy 🔢
- Scikit-learn 📊

---

## 📄 Licencia

📄 Distribuido bajo la [LICENSE MIT](LICENSE).

---

## 👨‍💻 Autor

**Ricardo Leitón**  
📧 ricardo.leiton@gmail.com

---

## 🧪 Cómo usar este proyecto

### 1. 🔃 Clonar el repositorio

Abrí una terminal y ejecutá:

```bash
git clone https://github.com/ricardoleiton/Lenguaje-de-senas-con-Python.git
cd Lenguaje-de-senas-con-Python
```

### 2. 📦 Instalar dependencias

Instalá las bibliotecas necesarias ejecutando:

```bash
pip install -r requirements.txt
```

O si no tenés ese archivo, podés instalarlas manualmente:

```bash
pip install opencv-python mediapipe numpy scikit-learn
```

---

### 3. 📸 Capturar datos

Ejecutá el script para capturar los landmarks de las manos (por letra):

```bash
python capturar_landmarks.py
```

Seguí las instrucciones en pantalla para grabar señas con tu cámara.

---

### 4. 🧠 Entrenar el modelo

Una vez tengas datos guardados en la carpeta `landmarks_data/`, ejecutá:

```bash
python entrenar_modelo.py
```

Esto generará un archivo `.pkl` con el modelo entrenado.

---

### 5. 🔍 Predecir señas en tiempo real

Probá el reconocimiento con la cámara ejecutando:

```bash
python predecir_todas_letras.py
```
