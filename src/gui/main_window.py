import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
from src.data_processing.loader import load_csv
from src.data_processing.preprocessing import preprocess_data, get_product_list
from src.forecasting.prophet_model import train_prophet


class MainWindow:

    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Demanda")
        self.root.geometry("620x480")

        self.csv_path = tk.StringVar()
        self.days = tk.IntVar(value=30)
        self.selected_product = tk.StringVar(value="— Global (todos los productos) —")

        self.df_raw = None
        self.df_processed = None
        self.model = None
        self.forecast = None

        self.create_widgets()

    def create_widgets(self):

        title = ttk.Label(
            self.root,
            text="Sistema de Predicción de Demanda",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        # --- Cargar CSV ---
        frame_file = ttk.Frame(self.root)
        frame_file.pack(fill="x", padx=20, pady=5)

        ttk.Button(
            frame_file,
            text="Cargar CSV",
            command=self.load_csv
        ).pack(side="left")

        ttk.Label(
            frame_file,
            textvariable=self.csv_path
        ).pack(side="left", padx=10)

        # --- Selector de producto ---
        frame_product = ttk.LabelFrame(self.root, text="Seleccionar Producto")
        frame_product.pack(fill="x", padx=20, pady=8)

        self.product_combobox = ttk.Combobox(
            frame_product,
            textvariable=self.selected_product,
            state="disabled",          # se habilita al cargar CSV
            width=55,
        )
        self.product_combobox.pack(padx=10, pady=6)

        # --- Días a predecir ---
        frame_days = ttk.Frame(self.root)
        frame_days.pack(fill="x", padx=20, pady=5)

        ttk.Label(frame_days, text="Días a predecir:").pack(side="left")
        ttk.Entry(frame_days, textvariable=self.days, width=10).pack(side="left", padx=10)

        # --- Botones principales ---
        frame_buttons = ttk.Frame(self.root)
        frame_buttons.pack(pady=12)

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

        # --- Métricas ---
        metrics_frame = ttk.LabelFrame(self.root, text="Métricas del modelo")
        metrics_frame.pack(fill="x", padx=20, pady=8)

        self.mae_label = ttk.Label(metrics_frame, text="MAE: ---")
        self.mae_label.pack(anchor="w", padx=10, pady=2)

        self.rmse_label = ttk.Label(metrics_frame, text="RMSE: ---")
        self.rmse_label.pack(anchor="w", padx=10, pady=2)

        self.product_label = ttk.Label(metrics_frame, text="Producto: ---")
        self.product_label.pack(anchor="w", padx=10, pady=2)

        # --- Exportar / Gráfica ---
        frame_export = ttk.Frame(self.root)
        frame_export.pack(pady=10)

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

    # ------------------------------------------------------------------ #

    def load_csv(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file:
            return
        try:
            self.df_raw = load_csv(file)
            self.csv_path.set(file)

            # Poblar el combobox con los productos del CSV
            productos = get_product_list(self.df_raw)
            self.product_combobox["values"] = productos
            self.product_combobox["state"] = "readonly"
            self.selected_product.set(productos[0])   # opción global por defecto

            # Resetear estado anterior
            self.df_processed = None
            self.model = None
            self.forecast = None
            self._reset_metrics()

            messagebox.showinfo(
                "Éxito",
                f"Archivo cargado\n\nFilas: {len(self.df_raw)}\n"
                f"Productos únicos: {len(productos) - 1}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def preprocess_data(self):
        if self.df_raw is None:
            messagebox.showwarning("Advertencia", "Primero cargue un CSV.")
            return
        try:
            product = self.selected_product.get()
            self.df_processed = preprocess_data(
                self.df_raw,
                date_column="Order Date",
                target_column="Sales",
                product_column="Product Name",
                product_name=product,
            )
            messagebox.showinfo(
                "Éxito",
                f"Datos procesados\n\n"
                f"Producto: {product}\n"
                f"Registros: {len(self.df_processed)}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def train_model(self):
        if self.df_processed is None:
            messagebox.showwarning("Advertencia", "Primero presione Preprocesar.")
            return
        try:
            product = self.selected_product.get()

            # Feedback visual mientras entrena
            self.root.config(cursor="wait")
            self.root.update()

            self.model, metrics = train_prophet(self.df_processed, product_name=product)

            # Actualizar etiquetas de métricas
            self.mae_label.config(text=f"MAE: {metrics['MAE']}")
            self.rmse_label.config(text=f"RMSE: {metrics['RMSE']}")
            self.product_label.config(text=f"Producto: {metrics['producto']}")

            messagebox.showinfo(
                "Modelo entrenado",
                f"Producto: {metrics['producto']}\n"
                f"Registros usados: {metrics['registros']}\n"
                f"MAE: {metrics['MAE']}\n"
                f"RMSE: {metrics['RMSE']}\n"
                f"Modelo guardado en: {metrics['model_path']}"
            )
        except Exception as e:
            messagebox.showerror("Error al entrenar", str(e))
        finally:
            self.root.config(cursor="")

    def predict(self):
        print("Generando predicción...")

    def show_graph(self):
        print("Mostrando gráfica...")

    def export_csv(self):
        print("Exportando resultados...")

    def _reset_metrics(self):
        self.mae_label.config(text="MAE: ---")
        self.rmse_label.config(text="RMSE: ---")
        self.product_label.config(text="Producto: ---")