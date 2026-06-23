import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
from src.data_processing.loader import load_csv
from src.data_processing.preprocessing import preprocess_data, get_product_list, CATEGORY_HEADER_PREFIX
from src.forecasting.prophet_model import train_prophet

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


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

        # --- Selector de producto/categoría ---
        frame_product = ttk.LabelFrame(self.root, text="Seleccionar Categoría / Producto")
        frame_product.pack(fill="x", padx=20, pady=8)

        self.product_combobox = ttk.Combobox(
            frame_product,
            textvariable=self.selected_product,
            state="disabled",
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

            items = get_product_list(self.df_raw)
            self.product_combobox["values"] = items
            self.product_combobox["state"] = "readonly"
            self.selected_product.set(items[0])

            self.df_processed = None
            self.model = None
            self.forecast = None
            self._reset_metrics()

            # Contar solo productos reales (excluir global y headers de categoría)
            n_productos = sum(
                1 for i in items
                if i != "— Global (todos los productos) —"
                and not i.startswith(CATEGORY_HEADER_PREFIX)
            )

            messagebox.showinfo(
                "Éxito",
                f"Archivo cargado\n\nFilas: {len(self.df_raw)}\n"
                f"Productos únicos: {n_productos}"
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

            # Etiqueta legible para el mensaje
            if product.startswith(CATEGORY_HEADER_PREFIX):
                label = f"Categoría: {product.replace(CATEGORY_HEADER_PREFIX, '').strip()}"
            elif product == "— Global (todos los productos) —":
                label = "Global (todos los productos)"
            else:
                label = f"Producto: {product}"

            messagebox.showinfo(
                "Éxito",
                f"Datos procesados\n\n"
                f"{label}\n"
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

            self.root.config(cursor="wait")
            self.root.update()

            self.model, metrics = train_prophet(self.df_processed, product_name=product)

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
        if self.df_raw is None:
            messagebox.showwarning("Sin datos", "Primero carga un archivo CSV.")
            return
        if self.model is None:
            messagebox.showwarning("Sin modelo", "Primero entrena el modelo con el botón 'Entrenar'.")
            return

        try:
            days = int(self.days.get())
            if days <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Días inválidos", "Ingresa un número entero positivo en 'Días a predecir'.")
            return

        try:
            self.root.config(cursor="wait")
            self.root.update()

            future = self.model.make_future_dataframe(periods=days, freq="D")
            self.forecast = self.model.predict(future)

        except Exception as e:
            messagebox.showerror("Error en predicción", str(e))
            return
        finally:
            self.root.config(cursor="")

        product = self.selected_product.get()
        messagebox.showinfo(
            "Predicción lista",
            f"Producto: {product}\n"
            f"Días predichos: {days}\n\n"
            "Usa 'Mostrar Gráfica' o 'Exportar CSV' para ver los resultados."
        )

    def show_graph(self):
        if self.forecast is None:
            messagebox.showwarning("Sin predicción", "Primero genera una predicción con el botón 'Predecir'.")
            return

        days = int(self.days.get())
        product = self.selected_product.get()

        win = tk.Toplevel(self.root)
        win.title(f"Gráfica — {product}")
        win.geometry("960x560")
        win.resizable(True, True)

        fig, ax = plt.subplots(figsize=(10, 4.8))
        fig.patch.set_facecolor("#F7F9FC")
        ax.set_facecolor("#F7F9FC")

        if self.df_processed is not None:
            ax.plot(
                self.df_processed["ds"],
                self.df_processed["y"],
                color="#4A90D9",
                linewidth=1.8,
                label="Histórico",
            )

        ax.fill_between(
            self.forecast["ds"],
            self.forecast["yhat_lower"],
            self.forecast["yhat_upper"],
            alpha=0.20,
            color="#F5A623",
            label="Intervalo de confianza",
        )

        ax.plot(
            self.forecast["ds"],
            self.forecast["yhat"],
            color="#E8572A",
            linewidth=2,
            linestyle="--",
            label="Predicción",
        )

        if len(self.forecast) > days:
            cutoff = self.forecast["ds"].iloc[-(days + 1)]
            ax.axvline(cutoff, color="#999", linewidth=1, linestyle=":")
            ymax = ax.get_ylim()[1]
            ax.text(cutoff, ymax * 0.97, "  Hoy", fontsize=8, color="#666", va="top")

        ax.set_title(f"Predicción de demanda — {product}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Fecha", fontsize=10)
        ax.set_ylabel("Ventas", fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)

        win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))

    def export_csv(self):
        if self.forecast is None:
            messagebox.showwarning("Sin predicción", "Primero genera una predicción con el botón 'Predecir'.")
            return

        days = int(self.days.get())
        product = self.selected_product.get()

        product_slug = (
            product.replace("— ", "").replace(" —", "")
            .replace(CATEGORY_HEADER_PREFIX, "")
            .replace(" ", "_").lower()
        )
        default_name = f"prediccion_{product_slug}_{days}d.csv"

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialdir=os.path.abspath("exports"),
            initialfile=default_name,
            title="Guardar predicción como…",
        )
        if not save_path:
            return

        cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        df_out = self.forecast[cols].tail(days).copy().reset_index(drop=True)
        df_out.rename(columns={
            "ds": "Fecha",
            "yhat": "Predicción",
            "yhat_lower": "Límite inferior",
            "yhat_upper": "Límite superior",
        }, inplace=True)
        df_out["Fecha"] = df_out["Fecha"].dt.strftime("%Y-%m-%d")

        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Exportación exitosa", f"Archivo guardado en:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))

    def _reset_metrics(self):
        self.mae_label.config(text="MAE: ---")
        self.rmse_label.config(text="RMSE: ---")
        self.product_label.config(text="Producto: ---")