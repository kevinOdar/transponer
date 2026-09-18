import io

import librosa
import numpy as np
import soundfile as sf
import streamlit as st

NOTAS = ["DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI"]

st.set_page_config(page_title="Transponer", page_icon="🎵")
st.title("🎵 Transponer canciones")
st.write("Sube un archivo de audio y cambia su tono (pitch) manteniendo el tempo.")

archivo = st.file_uploader("Archivo de audio", type=["wav", "mp3", "flac", "ogg", "m4a"])

modo = st.radio("Modo", ["Por nota", "Manual (semitonos)"], horizontal=True)

if modo == "Por nota":
    col1, col2 = st.columns(2)
    nota_origen = col1.selectbox("Tono actual", NOTAS, index=0)
    nota_destino = col2.selectbox("Tono deseado", NOTAS, index=11)
    diff = (NOTAS.index(nota_destino) - NOTAS.index(nota_origen)) % 12
    if diff > 6:
        diff -= 12
    semitonos = diff
    st.caption(f"Desplazamiento calculado: {semitonos:+d} semitonos")
else:
    semitonos = st.slider("Semitonos (+ sube, - baja)", -12, 12, 0)

if archivo is not None and st.button("Transponer", type="primary"):
    if semitonos == 0:
        st.warning("El desplazamiento es 0 semitonos, no hay nada que cambiar.")
    else:
        with st.spinner("Procesando audio..."):
            y, sr = librosa.load(archivo, sr=None, mono=False)
            if y.ndim == 1:
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitonos)
            else:
                y_shifted = np.stack(
                    [librosa.effects.pitch_shift(canal, sr=sr, n_steps=semitonos) for canal in y]
                )

            buffer = io.BytesIO()
            sf.write(buffer, y_shifted.T if y_shifted.ndim > 1 else y_shifted, sr, format="WAV")
            buffer.seek(0)

        st.success("¡Listo!")
        st.audio(buffer, format="audio/wav")
        st.download_button(
            "Descargar audio transpuesto",
            data=buffer,
            file_name="transpuesto.wav",
            mime="audio/wav",
        )
