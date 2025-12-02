# 🎵 Spotify Music Hub: Sistema de Recomendación y Análisis con IA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

Este proyecto es una aplicación integral de Ciencia de Datos y Machine Learning que analiza el dataset **"Top Canciones de Spotify 2023"**. El objetivo es descubrir patrones musicales, predecir géneros mediante modelos híbridos y ofrecer un sistema de recomendación personalizado a través de una interfaz web interactiva.

## 🚀 Características Principales

* **Dashboard Interactivo:** Una aplicación web construida con Streamlit que permite al usuario interactuar con los datos y modelos.
* **Sistema de Recomendación (KNN):** Sugiere las 5 canciones más similares a una elección del usuario basándose en características de audio.
* **Clasificación de Géneros (Híbrido):** Un enfoque innovador que combina GMM y Random Forest para determinar el género y subgénero musical.
* **Análisis Exploratorio:** Notebooks detallados con el proceso de limpieza, clustering y regresión.

---

## 🧠 Tecnología Aplicada y Metodología

El núcleo del proyecto se basa en tres cuadernos de Jupyter que alimentan la lógica de la aplicación:

### 1. Sistema de Recomendación (K-Nearest Neighbors)
Utilizamos el algoritmo **KNN** para calcular la distancia matemática entre vectores de características de audio (como *tempo*, *energía*, *bailaiblidad*, *valencia*).
* **Input:** Una canción seleccionada por el usuario.
* **Proceso:** Cálculo de distancia euclidiana en el espacio vectorial.
* **Output:** Las 5 "vecinas" más cercanas (canciones similares).

### 2. Gender Guessing (Modelo Híbrido)
Para la clasificación de géneros, no nos limitamos a un solo algoritmo. Implementamos un pipeline robusto:
1.  **GMM (Gaussian Mixture Models):** Para detectar agrupaciones naturales no supervisadas en los datos de audio.
2.  **Soft-Probability:** Asignación de probabilidades de pertenencia a múltiples géneros (evitando clasificaciones binarias rígidas).
3.  **Random Forest:** Un clasificador supervisado que toma las probabilidades anteriores para determinar el **Género** y **Subgénero** final con mayor precisión.

### 3. Predicción de Éxitos (Regresión)
*(Ubicado en `hit_prediction_regression.ipynb`)*
Análisis y modelos de regresión para intentar predecir la popularidad o el éxito de una canción basándose en sus atributos técnicos.

---

## 📂 Estructura del Proyecto

```text
├── 📓 gender_guessing_clustering.ipynb   # Modelado de géneros (GMM + RF)
├── 📓 hit_prediction_regression.ipynb    # Modelos de regresión para popularidad
├── 📓 songs_recomendation_system_knn.ipynb # Lógica del motor de recomendación
├── app.py                                # Archivo principal de Streamlit (Frontend)
├── data/                                 # Dataset de Spotify 2023
├── requirements.txt                      # Dependencias del proyecto
└── README.md                             # Documentación
```


### ⚙️ Cómo ejecutarlo

Sigue estos pasos para instalar las dependencias y correr la aplicación en tu entorno local:

**1. Instalación de librerías**

Abre tu terminal y ejecuta los siguientes comandos para instalar las herramientas necesarias:

```bash
pip install streamlit
pip install seaborn
pip install plotly
pip install pandas scikit-learn numpy matplotlib
```

## 2. Ejecutar la aplicación

Una vez instaladas las librerías, navega a la carpeta de la aplicación e iniciala:

```bash
cd app_streamlit
streamlit run home.py
```
