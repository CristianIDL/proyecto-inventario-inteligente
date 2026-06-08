import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
from src.data_processing.loader import load_csv
from src.data_processing.preprocessing import preprocess_data
class MainWindow:

    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Demanda")
        self.root.geometry("600x400")

        self.csv_path = tk.StringVar()
        self.days = tk.IntVar(value=30)
        self.df_raw = None
        self.df_processed = None
        self.create_widgets()

    def create_widgets(self):

        title = ttk.Label(
            self.root,
            text="Sistema de Predicción de Demanda",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        frame_file = ttk.Frame(self.root)
        frame_file.pack(fill="x", padx=20, pady=10)

        ttk.Button(
            frame_file,
            text="Cargar CSV",
            command=self.load_csv
        ).pack(side="left")

        ttk.Label(
            frame_file,
            textvariable=self.csv_path
        ).pack(side="left", padx=10)

        frame_days = ttk.Frame(self.root)
        frame_days.pack(fill="x", padx=20, pady=10)

        ttk.Label(
            frame_days,
            text="Días a predecir:"
        ).pack(side="left")

        ttk.Entry(
            frame_days,
            textvariable=self.days,
            width=10
        ).pack(side="left", padx=10)

        frame_buttons = ttk.Frame(self.root)
        frame_buttons.pack(pady=15)

        ttk.Button(
            frame_buttons,
            text="Preprocesar",
            command=self.preprocess_data
        ).grid(row=0, column=0, padx=5)

        ttk.Button(
            frame_buttons,
            text="Entrenar",
            command=self.train_model
        ).grid(row=0, column=1, padx=5)

        ttk.Button(
            frame_buttons,
            text="Predecir",
            command=self.predict
        ).grid(row=0, column=2, padx=5)

        metrics_frame = ttk.LabelFrame(
            self.root,
            text="Métricas"
        )
        metrics_frame.pack(fill="x", padx=20, pady=10)

        self.mae_label = ttk.Label(metrics_frame, text="MAE: ---")
        self.mae_label.pack(anchor="w", padx=10, pady=2)

        self.rmse_label = ttk.Label(metrics_frame, text="RMSE: ---")
        self.rmse_label.pack(anchor="w", padx=10, pady=2)

        frame_export = ttk.Frame(self.root)
        frame_export.pack(pady=20)

        ttk.Button(
            frame_export,
            text="Mostrar Gráfica",
            command=self.show_graph
        ).grid(row=0, column=0, padx=5)

        ttk.Button(
            frame_export,
            text="Exportar CSV",
            command=self.export_csv
        ).grid(row=0, column=1, padx=5)
    def load_csv(self):

        file = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )



        if not file:
            return

        try:
            self.df_raw = load_csv(file)

            self.df_raw = load_csv(file)

            print(type(self.df_raw))
            print(self.df_raw)
            print(self.df_raw.columns.tolist())
            self.csv_path.set(file)

            messagebox.showinfo(
                "Éxito",
                f"Archivo cargado\n\nFilas: {len(self.df_raw)}"
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                str(e)
            )

    def preprocess_data(self):

        if self.df_raw is None:
            messagebox.showwarning(
                "Advertencia",
                "Primero cargue un CSV."
            )
            return

        try:

            self.df_processed = preprocess_data(
                self.df_raw,
                date_column="Date",
                target_column="Sales"
            )

            messagebox.showinfo(
                "Éxito",
                f"Datos procesados\n\nRegistros: {len(self.df_processed)}"
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                str(e)
            )
    def train_model(self):
        print("Entrenando modelo...")

    def predict(self):
        print("Generando predicción...")

    def show_graph(self):
        print("Mostrando gráfica...")

    def export_csv(self):
        print("Exportando resultados...")