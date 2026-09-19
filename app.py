import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import audioread.exceptions
import librosa
import numpy as np
import soundfile as sf
import ttkbootstrap as ttk
import yt_dlp

CARPETA_DESCARGAS = os.path.join(tempfile.gettempdir(), "transponer_descargas")

NOTAS = ["DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI"]

# Perfiles tonales de Krumhansl-Schmuckler (Krumhansl & Kessler, 1982),
# usados para estimar la tonalidad correlacionando el croma promedio del
# audio contra estos perfiles rotados en las 12 tonalidades posibles.
PERFIL_MAYOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detectar_tono(ruta):
    y, sr = librosa.load(ruta, sr=None, mono=True)
    y_armonico = librosa.effects.harmonic(y)

    # El croma completo capta qué notas suenan, pero el croma del registro
    # grave capta la raíz real de cada acorde (el bajo), que es la señal más
    # fiable para distinguir una tonalidad de su dominante (p.ej. DO vs SOL,
    # que comparten 6 de 7 notas de la escala).
    croma = librosa.feature.chroma_cqt(y=y_armonico, sr=sr).mean(axis=1)
    croma_bajo = librosa.feature.chroma_cqt(
        y=y_armonico, sr=sr, fmin=librosa.note_to_hz("C1"), n_octaves=3
    ).mean(axis=1)
    perfil = croma + 1.5 * croma_bajo

    candidatos = []
    for i in range(12):
        candidatos.append((np.corrcoef(np.roll(PERFIL_MAYOR, i), perfil)[0, 1], i, "mayor"))
        candidatos.append((np.corrcoef(np.roll(PERFIL_MENOR, i), perfil)[0, 1], i, "menor"))
    candidatos.sort(key=lambda c: c[0], reverse=True)

    _, idx, modo = candidatos[0]
    alternativa = None
    score_top, score_2 = candidatos[0][0], candidatos[1][0]
    if score_top - score_2 < 0.03:
        alternativa = (candidatos[1][1], candidatos[1][2])
    return idx, modo, alternativa


# Plantillas de acorde: qué grados suenan sobre la fundamental (índice 0).
TEMPLATE_MAYOR = np.zeros(12)
TEMPLATE_MAYOR[[0, 4, 7]] = 1  # fundamental, 3ra mayor, 5ta justa
TEMPLATE_MENOR = np.zeros(12)
TEMPLATE_MENOR[[0, 3, 7]] = 1  # fundamental, 3ra menor, 5ta justa


NUMEROS_EN_PALABRAS = [
    "cero", "un", "dos", "tres", "cuatro", "cinco", "seis",
    "siete", "ocho", "nueve", "diez", "once", "doce",
]


def describir_semitonos(semitonos):
    cantidad = abs(semitonos)
    palabra = NUMEROS_EN_PALABRAS[cantidad] if cantidad < len(NUMEROS_EN_PALABRAS) else str(cantidad)
    unidad = "semitono" if cantidad == 1 else "semitonos"
    sentido = "más" if semitonos > 0 else "menos"
    return f"{palabra} {unidad} {sentido}"


def formatear_tiempo(segundos):
    minutos, s = divmod(int(segundos), 60)
    return f"{minutos}:{s:02d}"


def _suavizar_moda(etiquetas, ventana=5):
    n = len(etiquetas)
    medio = ventana // 2
    suavizado = []
    for i in range(n):
        vecinos = etiquetas[max(0, i - medio) : min(n, i + medio + 1)]
        conteo = {}
        for e in vecinos:
            conteo[e] = conteo.get(e, 0) + 1
        # ante empate, prioriza mantener la etiqueta actual (evita saltos innecesarios)
        suavizado.append(max(conteo.items(), key=lambda kv: (kv[1], kv[0] == etiquetas[i]))[0])
    return suavizado


def _fusionar_cambios_breves(segmentos, duracion_total, duracion_minima=3.0):
    if not segmentos:
        return segmentos
    resultado = [segmentos[0]]
    for i in range(1, len(segmentos)):
        tiempo, nombre = segmentos[i]
        siguiente = segmentos[i + 1][0] if i + 1 < len(segmentos) else duracion_total
        if siguiente - tiempo < duracion_minima or resultado[-1][1] == nombre:
            continue
        resultado.append((tiempo, nombre))
    return resultado


def detectar_acordes(ruta, ventana=1.5):
    y, sr = librosa.load(ruta, sr=None, mono=True)
    y_armonico = librosa.effects.harmonic(y)

    hop = 512
    croma = librosa.feature.chroma_cqt(y=y_armonico, sr=sr, hop_length=hop, norm=None)
    rms = librosa.feature.rms(y=y_armonico, hop_length=hop)[0]
    umbral = rms.max() * 0.08

    frames_por_bloque = max(1, round(sr / hop * ventana))
    n_bloques = croma.shape[1] // frames_por_bloque

    tiempos, etiquetas = [], []
    for b in range(n_bloques):
        ini, fin = b * frames_por_bloque, (b + 1) * frames_por_bloque
        tiempos.append(ini * hop / sr)
        if rms[ini:fin].mean() < umbral:
            etiquetas.append(None)
            continue

        vector = croma[:, ini:fin].mean(axis=1)
        vector = vector / (np.linalg.norm(vector) + 1e-9)

        mejor_score, mejor_nombre = -np.inf, None
        for i in range(12):
            score_mayor = np.dot(np.roll(TEMPLATE_MAYOR, i), vector)
            score_menor = np.dot(np.roll(TEMPLATE_MENOR, i), vector)
            if score_mayor > mejor_score:
                mejor_score, mejor_nombre = score_mayor, NOTAS[i]
            if score_menor > mejor_score:
                mejor_score, mejor_nombre = score_menor, f"{NOTAS[i]}m"
        etiquetas.append(mejor_nombre)

    # Una canción no cambia de acorde bloque a bloque: se suaviza con un
    # filtro de moda (descarta blips de 1-2 bloques) y luego se descartan
    # los cambios que no llegan a durar lo que dura un acorde real.
    etiquetas = _suavizar_moda(etiquetas)

    segmentos = []
    for tiempo, nombre in zip(tiempos, etiquetas):
        if segmentos and segmentos[-1][1] == nombre:
            continue
        segmentos.append((tiempo, nombre))

    duracion_total = len(y) / sr
    return _fusionar_cambios_breves(segmentos, duracion_total)


def descargar_audio_youtube(url, carpeta=CARPETA_DESCARGAS):
    os.makedirs(carpeta, exist_ok=True)
    opciones = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": os.path.join(carpeta, "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opciones) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def transponer_audio(ruta_entrada, semitonos, ruta_salida):
    y, sr = librosa.load(ruta_entrada, sr=None, mono=False)
    if y.ndim == 1:
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitonos)
    else:
        y_shifted = np.stack(
            [librosa.effects.pitch_shift(canal, sr=sr, n_steps=semitonos) for canal in y]
        )
    sf.write(ruta_salida, y_shifted.T if y_shifted.ndim > 1 else y_shifted, sr)


class App(ttk.Window):
    def __init__(self):
        super().__init__(
            title="Transponer",
            themename="darkly",
            resizable=(False, False),
            iconphoto=os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico"),
        )
        self.minsize(480, 0)
        self.ruta_archivo = None
        self.ruta_resultado = None
        self.tono_detectado_idx = None
        self.modo_detectado = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self, padding=24)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="🎵 Transponer", font=("Segoe UI", 20, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            main,
            text="Cambia el tono de una canción sin alterar el tempo",
            bootstyle="secondary",
        ).pack(anchor="w", pady=(0, 18))

        frame_archivo = ttk.Labelframe(main, text="Archivo", padding=14)
        frame_archivo.pack(fill="x", pady=(0, 14))
        fila_archivo = ttk.Frame(frame_archivo)
        fila_archivo.pack(fill="x")
        ttk.Button(
            fila_archivo,
            text="Elegir archivo...",
            bootstyle="primary",
            command=self._elegir_archivo,
        ).pack(side="left")
        self.label_archivo = ttk.Label(
            fila_archivo, text="Ningún archivo seleccionado", bootstyle="secondary"
        )
        self.label_archivo.pack(side="left", padx=12)

        fila_youtube = ttk.Frame(frame_archivo)
        fila_youtube.pack(fill="x", pady=(10, 0))
        self.entry_youtube = ttk.Entry(fila_youtube)
        self.entry_youtube.pack(side="left", fill="x", expand=True)
        self.entry_youtube.bind("<Return>", lambda _e: self._on_descargar_youtube())
        self.boton_youtube = ttk.Button(
            fila_youtube,
            text="Descargar de YouTube",
            bootstyle="danger-outline",
            command=self._on_descargar_youtube,
        )
        self.boton_youtube.pack(side="left", padx=(8, 0))

        frame_tono = ttk.Labelframe(main, text="Herramientas", padding=14)
        frame_tono.pack(fill="x", pady=(0, 18))

        self.notebook = ttk.Notebook(frame_tono)
        self.notebook.pack(fill="x")

        tab_nota = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(tab_nota, text="Detectar tono")
        self.label_tono_detectado = ttk.Label(
            tab_nota, text="Elige un archivo para detectar el tono", bootstyle="secondary"
        )
        self.label_tono_detectado.grid(row=0, column=0, columnspan=2, pady=(0, 12), sticky="w")
        ttk.Label(tab_nota, text="Transponer a").grid(row=1, column=0, padx=6, sticky="w")
        self.combo_destino = ttk.Combobox(tab_nota, values=NOTAS, state="readonly", width=8)
        self.combo_destino.current(11)
        self.combo_destino.grid(row=2, column=0, padx=6, pady=(4, 0), sticky="w")

        tab_manual = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(tab_manual, text="Manual")
        self.var_semitonos = tk.IntVar(value=0)
        fila_semitonos = ttk.Frame(tab_manual)
        fila_semitonos.pack(pady=(2, 8))
        ttk.Button(
            fila_semitonos,
            text="−",
            width=3,
            bootstyle="info-outline",
            command=lambda: self._cambiar_semitonos(-1),
        ).pack(side="left")
        self.label_semitonos = ttk.Label(
            fila_semitonos,
            text="0 semitonos",
            font=("Segoe UI", 11, "bold"),
            width=14,
            anchor="center",
        )
        self.label_semitonos.pack(side="left", padx=12)
        ttk.Button(
            fila_semitonos,
            text="+",
            width=3,
            bootstyle="info-outline",
            command=lambda: self._cambiar_semitonos(1),
        ).pack(side="left")
        ttk.Scale(
            tab_manual,
            from_=-12,
            to=12,
            orient="horizontal",
            variable=self.var_semitonos,
            command=self._actualizar_label_semitonos,
            bootstyle="info",
        ).pack(fill="x", padx=4)

        tab_acordes = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(tab_acordes, text="Acordes")
        self.boton_acordes = ttk.Button(
            tab_acordes,
            text="Detectar acordes",
            bootstyle="info-outline",
            command=self._on_detectar_acordes,
        )
        self.boton_acordes.pack(anchor="w")
        self.label_estado_acordes = ttk.Label(tab_acordes, text="", bootstyle="secondary")
        self.label_estado_acordes.pack(anchor="w", pady=(8, 8))

        frame_lista = ttk.Frame(tab_acordes)
        frame_lista.pack(fill="both", expand=True)
        self.lista_acordes = ttk.Treeview(
            frame_lista, columns=("tiempo", "acorde"), show="headings", height=8
        )
        self.lista_acordes.heading("tiempo", text="Tiempo")
        self.lista_acordes.heading("acorde", text="Acorde")
        self.lista_acordes.column("tiempo", width=70, anchor="center")
        self.lista_acordes.column("acorde", width=100, anchor="center")
        scrollbar_acordes = ttk.Scrollbar(
            frame_lista, orient="vertical", command=self.lista_acordes.yview
        )
        self.lista_acordes.configure(yscrollcommand=scrollbar_acordes.set)
        self.lista_acordes.pack(side="left", fill="both", expand=True)
        scrollbar_acordes.pack(side="right", fill="y")

        self.boton_transponer = ttk.Button(
            main,
            text="Transponer",
            bootstyle="success",
            command=self._on_transponer,
        )
        self.boton_transponer.pack(fill="x", ipady=6, pady=(0, 12))

        self.progress = ttk.Progressbar(main, mode="indeterminate", bootstyle="success-striped")

        self.label_estado = ttk.Label(main, text="", bootstyle="secondary", wraplength=420)
        self.label_estado.pack(fill="x", pady=(0, 8))

        self.boton_reproducir = ttk.Button(
            main,
            text="Abrir resultado",
            bootstyle="outline-secondary",
            command=self._abrir_resultado,
            state="disabled",
        )
        self.boton_reproducir.pack(fill="x")

    def _actualizar_label_semitonos(self, _valor=None):
        self.label_semitonos.config(text=f"{round(self.var_semitonos.get())} semitonos")

    def _cambiar_semitonos(self, delta):
        nuevo = max(-12, min(12, round(self.var_semitonos.get()) + delta))
        self.var_semitonos.set(nuevo)
        self._actualizar_label_semitonos()

    def _elegir_archivo(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona un archivo de audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if ruta:
            self._cargar_archivo(ruta)

    def _cargar_archivo(self, ruta):
        self.ruta_archivo = ruta
        self.label_archivo.config(text=os.path.basename(ruta), bootstyle="default")
        self.boton_reproducir.config(state="disabled")
        self.ruta_resultado = None
        self.label_estado.config(text="", bootstyle="secondary")
        self._iniciar_deteccion_tono(ruta)

    def _on_descargar_youtube(self):
        url = self.entry_youtube.get().strip()
        if not url:
            messagebox.showwarning("Falta el link", "Pega primero un link de YouTube.")
            return

        self.boton_youtube.config(state="disabled")
        self.label_archivo.config(text="Descargando de YouTube...", bootstyle="info")
        hilo = threading.Thread(target=self._descargar_youtube, args=(url,), daemon=True)
        hilo.start()

    def _descargar_youtube(self, url):
        try:
            ruta = descargar_audio_youtube(url)
        except Exception as exc:
            self.after(0, self._on_error_youtube, self._describir_error(exc))
        else:
            self.after(0, self._on_youtube_descargado, ruta)

    def _on_youtube_descargado(self, ruta):
        self.boton_youtube.config(state="normal")
        self._cargar_archivo(ruta)

    def _on_error_youtube(self, mensaje):
        self.boton_youtube.config(state="normal")
        self.label_archivo.config(text="No se pudo descargar el video", bootstyle="danger")
        messagebox.showerror("Error al descargar de YouTube", mensaje)

    def _iniciar_deteccion_tono(self, ruta):
        self.tono_detectado_idx = None
        self.modo_detectado = None
        self.label_tono_detectado.config(text="Detectando tono...", bootstyle="info")
        hilo = threading.Thread(target=self._detectar_tono, args=(ruta,), daemon=True)
        hilo.start()

    def _detectar_tono(self, ruta):
        try:
            idx, modo, alternativa = detectar_tono(ruta)
        except Exception as exc:
            self.after(0, self._on_error_deteccion, self._describir_error(exc))
        else:
            self.after(0, self._on_tono_detectado, idx, modo, alternativa)

    def _on_tono_detectado(self, idx, modo, alternativa):
        self.tono_detectado_idx = idx
        self.modo_detectado = modo
        texto = f"Tono detectado: {NOTAS[idx]} {modo}"
        if alternativa:
            texto += f"\n(no muy seguro, podría ser {NOTAS[alternativa[0]]} {alternativa[1]})"
        self.label_tono_detectado.config(text=texto, bootstyle="success")

    def _on_error_deteccion(self, mensaje):
        self.label_tono_detectado.config(text="No se pudo detectar el tono", bootstyle="danger")
        messagebox.showerror("Error al detectar el tono", mensaje)

    def _on_detectar_acordes(self):
        if not self.ruta_archivo:
            messagebox.showwarning("Falta archivo", "Primero elige un archivo de audio.")
            return

        self.boton_acordes.config(state="disabled")
        self.lista_acordes.delete(*self.lista_acordes.get_children())
        self.label_estado_acordes.config(
            text="Detectando acordes... (puede tardar unos segundos)", bootstyle="info"
        )
        hilo = threading.Thread(
            target=self._procesar_acordes, args=(self.ruta_archivo,), daemon=True
        )
        hilo.start()

    def _procesar_acordes(self, ruta):
        try:
            segmentos = detectar_acordes(ruta)
        except Exception as exc:
            self.after(0, self._on_error_acordes, self._describir_error(exc))
        else:
            self.after(0, self._on_acordes_detectados, segmentos)

    def _on_acordes_detectados(self, segmentos):
        self.boton_acordes.config(state="normal")
        self.label_estado_acordes.config(
            text=f"{len(segmentos)} cambios de acorde detectados", bootstyle="success"
        )
        for tiempo, nombre in segmentos:
            self.lista_acordes.insert("", "end", values=(formatear_tiempo(tiempo), nombre or "—"))

    def _on_error_acordes(self, mensaje):
        self.boton_acordes.config(state="normal")
        self.label_estado_acordes.config(
            text="No se pudieron detectar los acordes.", bootstyle="danger"
        )
        messagebox.showerror("Error al detectar acordes", mensaje)

    def _calcular_semitonos(self):
        if self.notebook.index(self.notebook.select()) == 0:
            if self.tono_detectado_idx is None:
                return None
            destino = NOTAS.index(self.combo_destino.get())
            diff = (destino - self.tono_detectado_idx) % 12
            if diff > 6:
                diff -= 12
            return diff
        return round(self.var_semitonos.get())

    def _on_transponer(self):
        if not self.ruta_archivo:
            messagebox.showwarning("Falta archivo", "Primero elige un archivo de audio.")
            return

        semitonos = self._calcular_semitonos()
        if semitonos is None:
            messagebox.showwarning(
                "Tono no detectado", "Espera a que termine de detectarse el tono, o usa la pestaña Manual."
            )
            return
        if semitonos == 0:
            messagebox.showinfo("Sin cambios", "El desplazamiento es 0 semitonos.")
            return

        base, _ext = os.path.splitext(os.path.basename(self.ruta_archivo))
        ruta_salida = filedialog.asksaveasfilename(
            title="Guardar audio transpuesto",
            initialfile=f"{base} {describir_semitonos(semitonos)}.wav",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
        )
        if not ruta_salida:
            return

        self.boton_transponer.config(state="disabled")
        self.boton_reproducir.config(state="disabled")
        self.progress.pack(fill="x", pady=(0, 10))
        self.progress.start(10)
        self.label_estado.config(text="Procesando...", bootstyle="info")

        hilo = threading.Thread(
            target=self._procesar, args=(self.ruta_archivo, semitonos, ruta_salida), daemon=True
        )
        hilo.start()

    def _procesar(self, ruta_entrada, semitonos, ruta_salida):
        try:
            transponer_audio(ruta_entrada, semitonos, ruta_salida)
        except Exception as exc:
            self.after(0, self._on_error, self._describir_error(exc))
        else:
            self.after(0, self._on_exito, ruta_salida)

    @staticmethod
    def _describir_error(exc):
        if isinstance(exc, audioread.exceptions.NoBackendError):
            return (
                "No se pudo leer este formato de audio.\n\n"
                "Necesitas tener ffmpeg instalado y en el PATH para leer "
                "mp3/m4a. Si lo acabas de instalar, reinicia la app."
            )
        mensaje = str(exc)
        return mensaje if mensaje else type(exc).__name__

    def _on_exito(self, ruta_salida):
        self.ruta_resultado = ruta_salida
        self.progress.stop()
        self.progress.pack_forget()
        self.boton_transponer.config(state="normal")
        self.boton_reproducir.config(state="normal")
        self.label_estado.config(text=f"Listo: {ruta_salida}", bootstyle="success")

    def _on_error(self, mensaje):
        self.progress.stop()
        self.progress.pack_forget()
        self.boton_transponer.config(state="normal")
        self.label_estado.config(text="Error al procesar el audio.", bootstyle="danger")
        messagebox.showerror("Error", mensaje)

    def _abrir_resultado(self):
        if self.ruta_resultado:
            os.startfile(self.ruta_resultado)


if __name__ == "__main__":
    App().mainloop()
