# Transponer

App de escritorio para transponer (cambiar el tono) de archivos de audio
manteniendo el tempo.

## Uso

```
.venv\Scripts\pip install -r requirements.txt   # solo la primera vez
.venv\Scripts\python app.py
```

Se abre una ventana. Elige el archivo de audio, el tono de origen y destino
(o el número de semitonos manualmente), pulsa "Transponer" y elige dónde
guardar el resultado.

Formatos de entrada soportados: wav, mp3, flac, ogg, m4a (mp3/m4a requieren
tener [ffmpeg](https://ffmpeg.org/) instalado y en el PATH). El resultado se
guarda siempre en WAV.
