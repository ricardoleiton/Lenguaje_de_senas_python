# 🤟 Lenguaje de Señas con Python

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow)

Este proyecto permite capturar, entrenar y predecir señas usando Python y MediaPipe.  
Ideal para aplicaciones educativas o inclusivas en el reconocimiento del lenguaje de señas.

---

## 📁 Estructura del Proyecto

```
├── modelo/
│   ├── knn_senas_estaticas.pkl
│   ├── lstm_senas_dinamicas.h5
│   └── senas_dinamicas_labels.pkl
├── senas_estaticas/
├── senas_dinamicas/
├── capturar_senas_estaticas.py
├── capturar_senas_dinamicas.py
├── entrenar_senas_estaticas_modelo.py
├── entrenar_senas_dinamicas_modelo.py
├── predecir_senas_estaticas.py
├── predecir_senas_dinamicas.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Tecnologías Utilizadas

- Python 🐍
- OpenCV 🎥
- MediaPipe ✋
- NumPy 🔢
- Scikit-learn 📊
- TensorFlow 🧠

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

```bash
git clone https://github.com/ricardoleiton/Lenguaje-de-senas-con-Python.git
cd Lenguaje-de-senas-con-Python
```

### 2. 📦 Instalar dependencias

```bash
pip install -r requirements.txt
```

O instalarlas manualmente:

```bash
pip install opencv-python mediapipe numpy scikit-learn tensorflow
```

---

### 3. 📸 Capturar señas

#### ✋ Estáticas (letras de la A a la Z y números del 0 al 9)
```bash
python capturar_senas_estaticas.py
```
- Ingresá una letra o número.
- Posicioná tu mano frente a la cámara.
- Presioná **"c"** para comenzar o **"q"** para cancelar.

#### 🤚 Dinámicas (palabras o frases en movimiento)
```bash
python capturar_senas_dinamicas.py
```
- Ingresá una palabra/frase sin espacios.
- Posicioná tu mano y presioná **"c"** para comenzar o **"q"** para cancelar.

---

### 4. 🧠 Entrenar los modelos

#### Para señas estáticas:
```bash
python entrenar_senas_estaticas_modelo.py
```

#### Para gestos dinámicos:
```bash
python entrenar_senas_dinamicas_modelo.py
```

---

### 5. 🔍 Predicción en tiempo real

#### Estáticas:
```bash
python predecir_senas_estaticas.py
```

#### Dinámicas:
```bash
python predecir_senas_dinamicas.py
```

---
