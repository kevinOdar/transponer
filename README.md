# Transponer

App para transponer (cambiar el tono) de archivos de audio manteniendo el tempo.

## Uso

```
.venv\Scripts\pip install -r requirements.txt   # solo la primera vez
.venv\Scripts\streamlit run app.py
```

Se abre en el navegador. Sube un audio, elige el tono de origen y destino (o el
número de semitonos manualmente), y descarga el resultado.

Formatos soportados: wav, mp3, flac, ogg, m4a (mp3/m4a requieren tener
[ffmpeg](https://ffmpeg.org/) instalado y en el PATH).
