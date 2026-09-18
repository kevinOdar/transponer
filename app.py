import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import audioread.exceptions
import librosa
import numpy as np
import soundfile as sf

NOTAS = ["DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI"]


def transponer_audio(ruta_entrada, semitonos, ruta_salida):
    y, sr = librosa.load(ruta_entrada, sr=None, mono=False)
    if y.ndim == 1:
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitonos)
    else:
        y_shifted = np.stack(
            [librosa.effects.pitch_shift(canal, sr=sr, n_steps=semitonos) for canal in y]
        )
    sf.write(ruta_salida, y_shifted.T if y_shifted.ndim > 1 else y_shifted, sr)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transponer")
        self.resizable(False, False)
        self.ruta_archivo = None
        self.ruta_resultado = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frame_archivo = ttk.Frame(self)
        frame_archivo.pack(fill="x", **pad)
        ttk.Button(frame_archivo, text="Elegir archivo...", command=self._elegir_archivo).pack(
            side="left"
        )
        self.label_archivo = ttk.Label(frame_archivo, text="Ningún archivo seleccionado")
        self.label_archivo.pack(side="left", padx=8)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="x", **pad)

        tab_nota = ttk.Frame(self.notebook)
        self.notebook.add(tab_nota, text="Por nota")
        ttk.Label(tab_nota, text="Tono actual:").grid(row=0, column=0, padx=5, pady=8, sticky="e")
        self.combo_origen = ttk.Combobox(tab_nota, values=NOTAS, state="readonly", width=8)
        self.combo_origen.current(0)
        self.combo_origen.grid(row=0, column=1, padx=5, pady=8)
        ttk.Label(tab_nota, text="Tono deseado:").grid(row=0, column=2, padx=5, pady=8, sticky="e")
        self.combo_destino = ttk.Combobox(tab_nota, values=NOTAS, state="readonly", width=8)
        self.combo_destino.current(11)
        self.combo_destino.grid(row=0, column=3, padx=5, pady=8)

        tab_manual = ttk.Frame(self.notebook)
        self.notebook.add(tab_manual, text="Manual (semitonos)")
        self.var_semitonos = tk.IntVar(value=0)
        self.label_semitonos = ttk.Label(tab_manual, text="0 semitonos")
        self.label_semitonos.pack(pady=(10, 0))
        ttk.Scale(
            tab_manual,
            from_=-12,
            to=12,
            orient="horizontal",
            variable=self.var_semitonos,
            command=self._actualizar_label_semitonos,
            length=280,
        ).pack(padx=10, pady=(0, 10))

        self.boton_transponer = ttk.Button(
            self, text="Transponer", command=self._on_transponer
        )
        self.boton_transponer.pack(pady=8)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=280)

        self.label_estado = ttk.Label(self, text="")
        self.label_estado.pack(**pad)

        self.boton_reproducir = ttk.Button(
            self, text="Abrir resultado", command=self._abrir_resultado, state="disabled"
        )
        self.boton_reproducir.pack(pady=(0, 10))

    def _actualizar_label_semitonos(self, _valor):
        self.label_semitonos.config(text=f"{self.var_semitonos.get()} semitonos")

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
            self.label_archivo.config(text=os.path.basename(ruta))
            self.boton_reproducir.config(state="disabled")
            self.ruta_resultado = None

    def _calcular_semitonos(self):
        if self.notebook.index(self.notebook.select()) == 0:
            origen = NOTAS.index(self.combo_origen.get())
            destino = NOTAS.index(self.combo_destino.get())
            diff = (destino - origen) % 12
            if diff > 6:
                diff -= 12
            return diff
        return self.var_semitonos.get()

    def _on_transponer(self):
        if not self.ruta_archivo:
            messagebox.showwarning("Falta archivo", "Primero elige un archivo de audio.")
            return

        semitonos = self._calcular_semitonos()
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
        self.progress.pack(pady=(0, 6))
        self.progress.start(10)
        self.label_estado.config(text="Procesando...")

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
        self.label_estado.config(text=f"Listo: {ruta_salida}")

    def _on_error(self, mensaje):
        self.progress.stop()
        self.progress.pack_forget()
        self.boton_transponer.config(state="normal")
        self.label_estado.config(text="Error al procesar el audio.")
        messagebox.showerror("Error", mensaje)

    def _abrir_resultado(self):
        if self.ruta_resultado:
            os.startfile(self.ruta_resultado)


if __name__ == "__main__":
    App().mainloop()
