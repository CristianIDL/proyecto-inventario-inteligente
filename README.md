# proyecto-inventario-inteligente


# Estructura del programa archivos
inventario_inteligente/
│
├── venv/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── exports/
│
├── models/
│
├── src/
│   ├── gui/
│   │   ├── main_window.py
│   │   └── widgets.py
│   │
│   ├── data_processing/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   ├── forecasting/
│   │   ├── prophet_model.py
│   │   └── metrics.py
│   │
│   └── utils/
│       └── file_manager.py
│
├── main.py
├── requirements.txt
└── README.md

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
