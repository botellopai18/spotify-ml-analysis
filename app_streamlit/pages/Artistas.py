import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuración de la página ---
st.set_page_config(page_title="Explorador de Artistas", layout="wide", page_icon="🎤")

# --- 1. CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Cargamos el mismo CSV actualizado
    df = pd.read_csv('df_songs_all_con_genero_subgenero.csv')
    df.columns = df.columns.str.strip()
    
    # Limpieza de streams (conversión a numérico)
    if 'streams' in df.columns and df['streams'].dtype == 'object':
        df['streams'] = df['streams'].astype(str).str.replace(',', '')
        df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error cargando CSV: {e}")
    st.stop()

# --- 2. LOGICA DE ARTISTAS ---
# Extraemos una lista única de TODOS los artistas individuales
# (Separando colaboraciones como "Drake, 21 Savage" en "Drake" y "21 Savage")
@st.cache_data
def obtener_lista_artistas(df):
    # Separamos por coma y eliminamos espacios extra
    todos_artistas = df['artist(s)_name'].str.split(',').explode().str.strip()
    # Obtenemos únicos y ordenamos
    lista = sorted(todos_artistas.unique())
    return lista

lista_artistas = obtener_lista_artistas(df)

# --- 3. SIDEBAR: BUSCADOR ---
st.sidebar.header("🔍 Buscar Artista")

# Usamos selectbox porque permite escribir para buscar
artista_seleccionado = st.sidebar.selectbox(
    "Escribe o selecciona un artista:",
    options=lista_artistas,
    index=None,
    placeholder="Ej. Bad Bunny, Taylor Swift..."
)

# --- 4. CONTENIDO PRINCIPAL ---
if artista_seleccionado:
    # FILTRADO INTELIGENTE:
    # Buscamos filas donde el artista seleccionado esté dentro de la lista de artistas de la canción
    # Esto asegura que si buscas "Drake", aparezca "Drake" y también "Drake, 21 Savage"
    mask = df['artist(s)_name'].apply(lambda x: artista_seleccionado in [a.strip() for a in str(x).split(',')])
    df_artista = df[mask]

    st.title(f"🎤 {artista_seleccionado}")

    # A. Métricas del Artista
    c1, c2, c3 = st.columns(3)
    
    total_streams = df_artista['streams'].sum()
    genero_top = df_artista['genre_inferred'].mode()[0] if not df_artista['genre_inferred'].empty else "N/A"
    
    c1.metric("Canciones en Top", len(df_artista))
    c2.metric("Total Streams", f"{total_streams:,.0f}") # Formato con comas
    c3.metric("Género Principal", genero_top)

    st.divider()

    # B. Gráfica: Versatilidad de Géneros
    # Mostramos qué géneros toca este artista
    if 'genre_inferred' in df_artista.columns:
        counts = df_artista['genre_inferred'].value_counts().reset_index()
        counts.columns = ['Género', 'Canciones']
        
        col_chart, col_empty = st.columns([1, 1]) # Usamos columnas para controlar el tamaño
        
        with col_chart:
            st.subheader("🎹 Versatilidad Musical")
            if not counts.empty:
                fig = px.pie(counts, names='Género', values='Canciones', hole=0.4, 
                             title=f"Géneros de {artista_seleccionado}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos suficientes de género.")

    # C. Lista de Canciones
    st.subheader(f"🎧 Discografía en el Dataset")
    
    cols_to_show = ['track_name', 'artist(s)_name', 'genre_inferred', 'subgenre_inferred', 'streams', 'year_released']
    # Aseguramos que las columnas existan
    cols = [c for c in cols_to_show if c in df_artista.columns]
    
    st.dataframe(
        df_artista[cols].sort_values('streams', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "streams": st.column_config.NumberColumn("Reproducciones", format="%d"),
            "year_released": st.column_config.NumberColumn("Año", format="%d"),
            "track_name": "Título",
            "artist(s)_name": "Artistas",
            "genre_inferred": "Género",
            "subgenre_inferred": "Subgénero"
        }
    )

else:
    # Pantalla de bienvenida si no hay selección
    st.info("👈 **Selecciona un artista** en la barra lateral para ver sus estadísticas y canciones.")
    st.write("### Artistas populares en la base de datos:")
    
    # Mostrar un top 10 rápido de artistas con más canciones para inspirar
    top_artistas = df['artist(s)_name'].str.split(',').explode().str.strip().value_counts().head(10)
    st.bar_chart(top_artistas)