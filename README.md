# proyecto-inventario-inteligente


# Estructura del programa archivos
# Estructura del Proyecto

```text
inventario_inteligente/
│
├── venv/                     # Entorno virtual de Python
│
├── data/                     # Datos utilizados por el sistema
│   ├── raw/                  # Datos originales sin procesar
│   └── processed/            # Datos limpios y transformados
│
├── exports/                  # Reportes, predicciones y archivos exportados
│
├── models/                   # Modelos entrenados y artefactos serializados
│
├── src/                      # Código fuente principal
│   │
│   ├── gui/                  # Interfaz gráfica de usuario
│   │   ├── main_window.py    # Ventana principal de la aplicación
│   │   └── widgets.py        # Componentes y controles personalizados
│   │
│   ├── data_processing/      # Procesamiento y preparación de datos
│   │   ├── loader.py         # Carga de archivos y datasets
│   │   └── preprocessing.py  # Limpieza y transformación de datos
│   │
│   ├── forecasting/          # Módulos de predicción
│   │   ├── prophet_model.py  # Implementación del modelo Prophet
│   │   └── metrics.py        # Métricas de evaluación
│   │
│   └── utils/                # Funciones auxiliares
│       └── file_manager.py   # Gestión de archivos y directorios
│
├── main.py                   # Punto de entrada de la aplicación
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Documentación principal
```

## Descripción General

**Inventario Inteligente** es una aplicación orientada al análisis y predicción de ventas mediante técnicas de series temporales. El sistema permite:

* Cargar y procesar información histórica de ventas.
* Limpiar y transformar datos para su análisis.
* Generar pronósticos utilizando Prophet.
* Evaluar el desempeño de los modelos predictivos.
* Exportar resultados y reportes.
* Interactuar con el sistema mediante una interfaz gráfica.


# Flujo de programa prototipo v 1.0 

Cargar CSV
    ↓
Preprocesar datos
    ↓
Entrenar Prophet
    ↓
Predecir N días
    ↓
Mostrar métricas
    ↓
Mostrar gráficas
    ↓
Exportar CSV

# Creacion de entorno de python para desarrollo 

py -m venv venv

.\venv\Scripts\activate

pip install pandas prophet matplotlib scikit-learn tkinterdnd2

pip freeze > requirements.txt


# Instalacioin de entorno de desarrollo
### 1. Ir a la carpeta del backend

```bash
cd proyecto-inventario-inteligente
```

### 2. Crear y activar un entorno virtual

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```
