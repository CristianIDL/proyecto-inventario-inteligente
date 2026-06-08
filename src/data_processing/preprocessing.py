import pandas as pd


def preprocess_data(df, date_column, target_column):

    # Copia para no modificar el original
    df = df.copy()

    # Eliminar filas vacías
    df = df.dropna()

    # Convertir fecha
    df[date_column] = pd.to_datetime(df[date_column])

    # Agrupar ventas por fecha
    df = (
        df.groupby(date_column)[target_column]
        .sum()
        .reset_index()
    )

    # Formato requerido por Prophet
    df = df.rename(
        columns={
            date_column: "ds",
            target_column: "y"
        }
    )

    # Ordenar cronológicamente
    df = df.sort_values("ds")

    return df