import pandas as pd

def get_product_list(df, product_column="Product Name"):
    """Retorna lista ordenada de productos únicos del DataFrame raw."""
    if product_column not in df.columns:
        raise ValueError(f"Columna '{product_column}' no encontrada en el CSV.")
    productos = sorted(df[product_column].dropna().unique().tolist())
    return ["— Global (todos los productos) —"] + productos


def preprocess_data(df, date_column, target_column, product_column=None, product_name=None):
    """
    Preprocesa el DataFrame para Prophet.
    Si product_name es None o 'global', agrega todas las ventas.
    Si product_name está definido, filtra por ese producto.
    """
    df = df.copy()

    # Filtrar por producto si se especifica
    if product_name and product_name != "— Global (todos los productos) —":
        if product_column not in df.columns:
            raise ValueError(f"Columna '{product_column}' no encontrada.")
        df = df[df[product_column] == product_name]
        if df.empty:
            raise ValueError(f"No hay datos para el producto: '{product_name}'")

    # Parsear fechas de forma robusta
    try:
        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
    except Exception:
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                df[date_column] = pd.to_datetime(df[date_column], format=fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"No se pudo parsear la columna de fechas '{date_column}'.")

    df = (
        df.groupby(date_column)[target_column]
        .sum()
        .reset_index()
    )

    df = df.rename(columns={date_column: "ds", target_column: "y"})
    df = df.sort_values("ds").reset_index(drop=True)

    return df