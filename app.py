import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import audioread.exceptions
import librosa
import numpy as np
import soundfile as sf
import ttkbootstrap as ttk

NOTAS = ["DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI"]

# Perfiles tonales de Krumhansl-Schmuckler (Krumhansl & Kessler, 1982),
# usados para estimar la tonalidad correlacionando el croma promedio del
# audio contra estos perfiles rotados en las 12 tonalidades posibles.
PERFIL_MAYOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detectar_tono(ruta):
    y, sr = librosa.load(ruta, sr=None, mono=True)
    croma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)

    mejor_score, mejor_idx, mejor_modo = -np.inf, 0, "mayor"
    for i in range(12):
        score_mayor = np.corrcoef(np.roll(PERFIL_MAYOR, i), croma)[0, 1]
        score_menor = np.corrcoef(np.roll(PERFIL_MENOR, i), croma)[0, 1]
        if score_mayor > mejor_score:
            mejor_score, mejor_idx, mejor_modo = score_mayor, i, "mayor"
        if score_menor > mejor_score:
            mejor_score, mejor_idx, mejor_modo = score_menor, i, "menor"
    return mejor_idx, mejor_modo


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
        super().__init__(title="Transponer", themename="darkly", resizable=(False, False))
        self.geometry("440x480")
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
        ttk.Button(
            frame_archivo,
            text="Elegir archivo...",
            bootstyle="primary",
            command=self._elegir_archivo,
        ).pack(side="left")
        self.label_archivo = ttk.Label(
            frame_archivo, text="Ningún archivo seleccionado", bootstyle="secondary"
        )
        self.label_archivo.pack(side="left", padx=12)

        frame_tono = ttk.Labelframe(main, text="Transposición", padding=14)
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
        self.label_semitonos = ttk.Label(
            tab_manual, text="0 semitonos", font=("Segoe UI", 10, "bold")
        )
        self.label_semitonos.pack(pady=(2, 6))
        ttk.Scale(
            tab_manual,
            from_=-12,
            to=12,
            orient="horizontal",
            variable=self.var_semitonos,
            command=self._actualizar_label_semitonos,
            bootstyle="info",
        ).pack(fill="x", padx=4)

        self.boton_transponer = ttk.Button(
            main,
            text="Transponer",
            bootstyle="success",
            command=self._on_transponer,
        )
        self.boton_transponer.pack(fill="x", ipady=6, pady=(0, 12))

        self.progress = ttk.Progressbar(main, mode="indeterminate", bootstyle="success-striped")

        self.label_estado = ttk.Label(main, text="", bootstyle="secondary", wraplength=380)
        self.label_estado.pack(fill="x", pady=(0, 8))

        self.boton_reproducir = ttk.Button(
            main,
            text="Abrir resultado",
            bootstyle="outline-secondary",
            command=self._abrir_resultado,
            state="disabled",
        )
        self.boton_reproducir.pack(fill="x")

    def _actualizar_label_semitonos(self, _valor):
        self.label_semitonos.config(text=f"{round(self.var_semitonos.get())} semitonos")

    def _elegir_archivo(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona un archivo de audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if ruta:
            self.ruta_archivo = ruta
            self.label_archivo.config(text=os.path.basename(ruta), bootstyle="default")
            self.boton_reproducir.config(state="disabled")
            self.ruta_resultado = None
            self.label_estado.config(text="", bootstyle="secondary")
            self._iniciar_deteccion_tono(ruta)

    def _iniciar_deteccion_tono(self, ruta):
        self.tono_detectado_idx = None
        self.modo_detectado = None
        self.label_tono_detectado.config(text="Detectando tono...", bootstyle="info")
        hilo = threading.Thread(target=self._detectar_tono, args=(ruta,), daemon=True)
        hilo.start()

    def _detectar_tono(self, ruta):
        try:
            idx, modo = detectar_tono(ruta)
        except Exception as exc:
            self.after(0, self._on_error_deteccion, self._describir_error(exc))
        else:
            self.after(0, self._on_tono_detectado, idx, modo)

    def _on_tono_detectado(self, idx, modo):
        self.tono_detectado_idx = idx
        self.modo_detectado = modo
        self.label_tono_detectado.config(
            text=f"Tono detectado: {NOTAS[idx]} {modo}", bootstyle="success"
        )

    def _on_error_deteccion(self, mensaje):
        self.label_tono_detectado.config(text="No se pudo detectar el tono", bootstyle="danger")
        messagebox.showerror("Error al detectar el tono", mensaje)

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
            initialfile=f"{base}_transpuesto.wav",
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
