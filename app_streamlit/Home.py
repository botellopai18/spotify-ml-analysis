import streamlit as st

# --- Configuración de la página ---
st.set_page_config(
    page_title="Inicio - Spotify 2023 Project",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Encabezado ---
st.write("# 🎹 Explorador y Recomendador Musical: Spotify Top 2023 🎧")
st.markdown("""---""")

# --- Introducción ---
st.markdown("""
### ¡Bienvenido a tu nuevo hub musical!

Esta aplicación utiliza el dataset **Top Canciones de Spotify 2023**. 
Nuestro objetivo es ayudarte a descubrir nueva música y entender las tendencias del año utilizando Inteligencia Artificial.
""")

# --- Sección Técnica ---
st.header("🧠 Tecnología aplicada")

col_tech1, col_tech2 = st.columns(2, gap="medium")

with col_tech1:
    with st.container(border=True): # Agregamos borde para que se vea mejor
        st.subheader("🤖 Género Guessing (Híbrido)")
        st.info("""
        Implementamos un sistema robusto de clasificación que combina tres técnicas clave:
        
        1.  **GMM (Gaussian Mixture Models):** Para encontrar agrupaciones naturales en los datos de audio.
        2.  **Soft-Probability:** Para asignar probabilidades de pertenencia a múltiples géneros (no solo binario).
        3.  **Random Forest:** El clasificador final que toma estas probabilidades y determina el **Género** y **Subgénero** más probable.
        """)

with col_tech2:
    with st.container(border=True):
        st.subheader("🔍 Recomendación de Canciones")
        st.success("""
        Nuestro motor de recomendación utiliza el algoritmo **KNN (K-Nearest Neighbors)**.
        
        Calculamos la distancia matemática entre las características de audio (tempo, energía, bailabilidad, etc.) de cada canción para sugerirte las **5 pistas más cercanas** a tu selección dentro de nuestro espacio vectorial.
        """)

st.divider()



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# --- Configuración de página ---
st.set_page_config(page_title="EDA - Spotify 2023", layout="wide", page_icon="📊")

st.title("📊 Análisis Exploratorio de Datos (EDA)")
st.markdown("Visión general de las métricas, correlaciones y tendencias del dataset.")

# --- Carga de Datos ---
@st.cache_data
def load_data():
    # Cargamos el CSV final
    df = pd.read_csv('df_songs_all_con_genero_subgenero.csv')
    
    # Conversión de tipos básica para que las gráficas funcionen
    # (No es limpieza de nulos, solo aseguramiento de tipos numéricos)
    if 'streams' in df.columns and df['streams'].dtype == 'object':
        df['streams'] = df['streams'].astype(str).str.replace(',', '')
        df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
        
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Falta el archivo 'df_songs_all_con_genero_subgenero.csv'")
    st.stop()

# --- 1. KPIs GENERALES ---
st.subheader("📌 Métricas Globales")
col1, col2, col3, col4 = st.columns(4)

# Cálculos
total_songs = len(df)
total_artists = df['artist(s)_name'].nunique()
total_genres = df['genre_inferred'].nunique() if 'genre_inferred' in df.columns else 0
avg_streams = df['streams'].mean()

col1.metric("Total Canciones", total_songs)
col2.metric("Artistas Únicos", total_artists)
col3.metric("Géneros Identificados", total_genres)
col4.metric("Promedio Reproducciones", f"{avg_streams/1e6:.1f} M")

st.divider()

# --- 2. PESTAÑAS DE ANÁLISIS ---
tab1, tab2, tab3 = st.tabs(["🏆 Rankings y Distribuciones", "🔥 Mapas de Calor (Heatmaps)", "📈 Relaciones"])

# === TAB 1: RANKINGS ===
with tab1:
    col_art, col_gen = st.columns(2)
    
    # A. Top 10 Artistas (por cantidad de canciones en el Top)
    with col_art:
        st.subheader("Top 10 Artistas (Más canciones)")
        # Separamos artistas por comas para contar individualmente
        all_artists = df['artist(s)_name'].str.split(',').explode().str.strip()
        top_artists = all_artists.value_counts().head(10).reset_index()
        top_artists.columns = ['Artista', 'Canciones']
        
        fig_art = px.bar(top_artists, x='Canciones', y='Artista', orientation='h', 
                         color='Canciones', color_continuous_scale='Viridis',
                         text_auto=True)
        fig_art.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_art, use_container_width=True)

    # B. Top Géneros Musicales
    with col_gen:
        st.subheader("Distribución de Géneros")
        if 'genre_inferred' in df.columns:
            top_genres = df['genre_inferred'].value_counts().reset_index()
            top_genres.columns = ['Género', 'Total']
            
            fig_gen = px.pie(top_genres, names='Género', values='Total', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_gen, use_container_width=True)
        else:
            st.warning("No se encontró la columna de género.")

# === TAB 2: HEATMAPS ===
with tab2:
    st.write("Análisis de correlaciones y relaciones categóricas.")
    
    c_heat1, c_heat2 = st.columns(2)

    # A. Heatmap Cuantitativo (Correlación Pearson)
    with c_heat1:
        st.subheader("🔥 Correlación: Audio Features")
        features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
        
        # Filtramos solo columnas que existen
        valid_feats = [f for f in features if f in df.columns]
        corr_matrix = df[valid_feats].corr()

        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    linewidths=0.5, ax=ax_corr, cbar_kws={"shrink": .8})
        st.pyplot(fig_corr)
        st.caption("Muestra qué características numéricas aumentan o disminuyen juntas.")

    

# === TAB 3: RELACIONES (SCATTERS) ===
with tab3:
    st.subheader("Impacto de Audio Features en Popularidad")
    
    col_sel1, col_sel2 = st.columns([1, 3])
    
    with col_sel1:
        feature_x = st.selectbox("Selecciona característica X:", 
                                 ['danceability_%', 'energy_%', 'valence_%', 'bpm'], index=0)
    
    with col_sel2:
        # Gráfico de dispersión: Streams vs Característica seleccionada
        # Coloreado por Género para ver agrupaciones
        fig_scat = px.scatter(df, x=feature_x, y='streams', 
                              color='genre_inferred' if 'genre_inferred' in df.columns else None,
                              size='in_spotify_playlists', # El tamaño es la presencia en playlists
                              hover_name='track_name',
                              log_y=True, # Escala logarítmica para ver mejor los streams
                              title=f"Streams vs {feature_x}",
                              height=500)
        st.plotly_chart(fig_scat, use_container_width=True)

st.markdown("---")
st.caption("Proyecto de Ciencia de Datos - Spotify 2023 Dataset")