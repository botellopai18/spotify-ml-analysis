import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Spotify Recommender Pro", layout="wide")
sns.set_style("whitegrid")

# --- 1. CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Asegúrate de que el nombre del archivo sea el correcto
    df = pd.read_csv('df_songs_all_con_genero_subgenero.csv') 
    
    # Limpieza básica
    df.columns = df.columns.str.strip()
    df['search_label'] = df['track_name'] + " - " + df['artist(s)_name']
    
    cols_to_numeric = [
        'id_song', 'id_rec_1', 'id_rec_2', 'id_rec_3', 'id_rec_4', 'id_rec_5', 
        'bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
        'instrumentalness_%', 'liveness_%', 'speechiness_%',
        'streams', 'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
        'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts'
    ]
    
    for col in cols_to_numeric:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

try:
    df_completo = load_data()
except FileNotFoundError:
    st.error("⚠️ No se encontró el archivo. Asegúrate de subir el CSV correcto.")
    st.stop()

# --- 2. FUNCIONES DE VISUALIZACIÓN (SIN CAMBIOS) ---

def evaluar_coherencia_visual(song_id, df_completo):
    features_to_check = ['bpm', 'energy_%', 'danceability_%', 'valence_%', 
                         'acousticness_%', "instrumentalness_%", "liveness_%", "speechiness_%"]
    fila_original = df_completo[df_completo['id_song'] == song_id]
    if fila_original.empty: return None, None

    original_stats = fila_original.iloc[0][features_to_check]
    nombre_cancion = fila_original.iloc[0]['track_name']
    cols_recs = ['id_rec_1', 'id_rec_2', 'id_rec_3', 'id_rec_4', 'id_rec_5']
    ids_vecinos = fila_original.iloc[0][cols_recs].dropna().values
    vecinos_df = df_completo[df_completo['id_song'].isin(ids_vecinos)]
    reco_stats = vecinos_df[features_to_check]
    
    reco_mean = reco_stats.mean()
    diff = np.abs(original_stats - reco_mean)

    comparativa = pd.DataFrame({
        'Original': original_stats,
        'Promedio Recs': reco_mean,
        'Diferencia (Abs)': diff,
        'Mínimo Recs': reco_stats.min(),
        'Máximo Recs': reco_stats.max()
    })

    indices_x = np.arange(len(features_to_check))
    width = 0.35
    fig = plt.figure(figsize=(14, 5))
    plt.bar(indices_x - width/2, original_stats, width, label='Original', color='#1DB954', alpha=0.9)
    plt.bar(indices_x + width/2, reco_mean, width, label='Promedio Recomendados', color='#191414', alpha=0.7)
    plt.title(f'Comparación de Audio Features: {nombre_cancion}', fontsize=14)
    plt.xticks(indices_x, features_to_check, rotation=15)
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return comparativa, fig

def grafica_similares_dos_caracteristicas_df_completo(idx_song, caracteristicas, df_completo):
    fila_original = df_completo[df_completo['id_song'] == idx_song]
    if fila_original.empty: return None

    x_col, y_col = caracteristicas
    x_origin = fila_original.iloc[0][x_col]
    y_origin = fila_original.iloc[0][y_col]
    nombre_origin = fila_original.iloc[0]['track_name']
    cols_recs = ['id_rec_1', 'id_rec_2', 'id_rec_3', 'id_rec_4', 'id_rec_5']
    ids_vecinos = fila_original.iloc[0][cols_recs].dropna().values
    vecinos_df = df_completo[df_completo['id_song'].isin(ids_vecinos)]

    fig = plt.figure(figsize=(10, 6)) 
    plt.scatter(df_completo[x_col], df_completo[y_col], c='lightgray', s=20, alpha=0.3, label='Resto del Dataset', zorder=0)
    for _, row in vecinos_df.iterrows():
        plt.plot([x_origin, row[x_col]], [y_origin, row[y_col]], c='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    plt.scatter(vecinos_df[x_col], vecinos_df[y_col], c='dodgerblue', s=100, edgecolors='white', alpha=0.9, label='Recomendaciones', zorder=2)
    plt.scatter(x_origin, y_origin, c='crimson', s=250, marker='*', edgecolors='black', label='Original', zorder=3)
    plt.text(x_origin, y_origin, f"  {nombre_origin}", fontsize=11, fontweight='bold', color='darkred', zorder=4, verticalalignment='bottom')
    for _, row in vecinos_df.iterrows():
        plt.text(row[x_col], row[y_col], f"  {row['track_name']}", fontsize=9, color='black', alpha=0.8, zorder=4)
    plt.title(f'Mapa de Similitud: {x_col} vs {y_col}', fontsize=14)
    plt.xlabel(x_col.capitalize(), fontsize=12)
    plt.ylabel(y_col.capitalize(), fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    return fig 

# --- 3. INTERFAZ PRINCIPAL ---

st.title("🎵 Dashboard de Recomendación Musical")

opcion = st.selectbox(
    "Selecciona una canción:",
    options=df_completo['search_label'].unique(),
    index=None,
    placeholder="Buscar..."
)

if opcion:
    song_row = df_completo[df_completo['search_label'] == opcion].iloc[0]
    id_seleccionado = song_row['id_song']
    
    # --- 1. INFO HEADER (Lógica Condicional para Subgénero) ---
    
    # Verificamos si existe el subgénero y no es NaN
    sub_val = song_row.get('subgenre_inferred')
    has_subgenre = pd.notna(sub_val) and str(sub_val).strip() != ''

    if has_subgenre:
        # Si tiene subgénero, usamos 3 columnas
        col_info, col_gen, col_sub = st.columns([2, 1, 1])
    else:
        # Si no tiene, usamos 2 columnas (más espacio para info)
        col_info, col_gen = st.columns([3, 1])

    with col_info:
        st.markdown(f"## 🎵 {song_row['track_name']}")
        st.markdown(f"**Artista:** {song_row['artist(s)_name']}")
    
    with col_gen:
        st.metric(label="Género", value=song_row.get('genre_inferred', 'N/A'))
        
    if has_subgenre:
        with col_sub:
            st.metric(label="Subgénero", value=sub_val)
    
    st.divider()

    # --- 2. RECOMENDACIONES (Lógica Condicional en Texto) ---
    st.subheader(f"🎧 Si te gusta, escucha esto:")
    
    rec_ids = [song_row[f'id_rec_{i}'] for i in range(1, 6)]
    recs_df = df_completo[df_completo['id_song'].isin(rec_ids)]
    
    cols = st.columns(5)
    for idx, (i, row) in enumerate(recs_df.iterrows()):
        with cols[idx]:
            # Construimos el texto del género dinámicamente
            genre_text = row.get('genre_inferred', '')
            sub_text = row.get('subgenre_inferred')
            
            # Solo agregamos el subgénero si es válido (no es NaN)
            if pd.notna(sub_text) and str(sub_text).strip() != '':
                display_genre = f"🎼 {genre_text} • {sub_text}"
            else:
                display_genre = f"🎼 {genre_text}"
                
            st.info(f"**{row['track_name']}**\n\n*{row['artist(s)_name']}*\n\n{display_genre}")

    st.divider()

    # --- SECCIÓN POPULARIDAD (SIN CAMBIOS) ---
    st.subheader("🔥 Popularidad e Impacto")
    
    col_streams, col_spotify, col_charts = st.columns(3)
    
    with col_streams:
        streams_val = song_row['streams']
        st.metric("Reproducciones Totales (Streams)", f"{int(streams_val):,}")
        st.caption("Total acumulado en Spotify 2023")

    with col_spotify:
        st.markdown("##### 🟢 Presencia en Spotify")
        st.write(f"📂 En **{int(song_row['in_spotify_playlists'])}** Playlists")
        st.write(f"📈 En **{int(song_row['in_spotify_charts'])}** Charts")

    with col_charts:
        st.markdown("##### 🌍 Otras Plataformas (Charts)")
        col_a, col_d, col_s = st.columns(3)
        col_a.metric("Apple", int(song_row['in_apple_charts']))
        col_d.metric("Deezer", int(song_row['in_deezer_charts']))
        col_s.metric("Shazam", int(song_row['in_shazam_charts']))

    st.divider()

    # --- ANÁLISIS DE COHERENCIA (SIN CAMBIOS) ---
    st.subheader("📊 Análisis de Coherencia")
    st.write("Comparamos las características de la canción original vs. el promedio de las 5 recomendaciones.")
    
    df_tabla, fig_barras = evaluar_coherencia_visual(id_seleccionado, df_completo)
    
    if df_tabla is not None:
        tab1, tab2 = st.tabs(["📈 Gráfico Comparativo", "📋 Tabla de Datos"])
        with tab1:
            st.pyplot(fig_barras)
        with tab2:
            st.dataframe(
                df_tabla.style.format("{:.2f}").background_gradient(cmap="Blues", subset=['Diferencia (Abs)']),
                use_container_width=True
            )

    st.divider()

    # --- MAPA DE SIMILITUD (SIN CAMBIOS) ---
    st.subheader("🗺️ Mapa Visual de Similitud")
    st.write("Compara la canción seleccionada con sus recomendaciones en el mapa global.")

    col_x, col_y = st.columns(2)
    audio_features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                      'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
    
    with col_x:
        eje_x = st.selectbox("Eje X:", options=audio_features, index=6) 
    with col_y:
        eje_y = st.selectbox("Eje Y:", options=audio_features, index=3) 

    if eje_x and eje_y:
        figura = grafica_similares_dos_caracteristicas_df_completo(id_seleccionado, [eje_x, eje_y], df_completo)
        if figura:
            st.pyplot(figura)
        else:
            st.warning("No se pudo generar la gráfica.")