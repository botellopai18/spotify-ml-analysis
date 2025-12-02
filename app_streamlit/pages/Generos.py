import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Explorador Géneros", layout="wide")

@st.cache_data
def load_data():
    # NOMBRE DE ARCHIVO ACTUALIZADO
    df = pd.read_csv('df_songs_all_con_genero_subgenero.csv')
    df.columns = df.columns.str.strip()
    
    # Limpieza numérica streams
    if 'streams' in df.columns and df['streams'].dtype == 'object':
        df['streams'] = df['streams'].astype(str).str.replace(',', '')
        df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
        
    return df

try:
    df = load_data()
except:
    st.error("Error cargando CSV")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("🔍 Filtros")
# Usamos 'genre_inferred'
lista = sorted(df['genre_inferred'].dropna().unique())
sel_gen = st.sidebar.radio("Género:", lista)

# --- CONTENIDO ---
df_g = df[df['genre_inferred'] == sel_gen]

st.title(f"🎼 {sel_gen}")

c1, c2, c3 = st.columns(3)
c1.metric("Canciones", len(df_g))
c2.metric("Artistas", df_g['artist(s)_name'].nunique())
c3.metric("BPM Promedio", int(df_g['bpm'].mean()))

st.divider()

# Gráfico Subgéneros
if 'subgenre_inferred' in df_g.columns:
    # Contamos ignorando nulos
    counts = df_g['subgenre_inferred'].dropna().value_counts().reset_index()
    counts.columns = ['Subgénero', 'Total']
    
    if not counts.empty:
        st.subheader("Distribución de Subgéneros")
        fig = px.pie(counts, names='Subgénero', values='Total', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Lista de Canciones")
cols = ['track_name', 'artist(s)_name', 'subgenre_inferred', 'streams']
st.dataframe(
    df_g[cols].sort_values('streams', ascending=False),
    use_container_width=True,
    hide_index=True
)