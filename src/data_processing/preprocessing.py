import pandas as pd

CATEGORY_HEADER_PREFIX = "── "

def get_product_list(df, product_column="Product Name", category_column="Category"):
    """
    Retorna lista combinada: Global + categorías como headers + productos anidados.
    """
    if product_column not in df.columns:
        raise ValueError(f"Columna '{product_column}' no encontrada en el CSV.")

    items = ["— Global (todos los productos) —"]

    if category_column in df.columns:
        categorias = sorted(df[category_column].dropna().unique().tolist())
        for cat in categorias:
            items.append(f"{CATEGORY_HEADER_PREFIX}{cat}")
            productos = sorted(
                df[df[category_column] == cat][product_column]
                .dropna().unique().tolist()
            )
            items.extend(productos)
    else:
        productos = sorted(df[product_column].dropna().unique().tolist())
        items.extend(productos)

    return items


def preprocess_data(
    df,
    date_column,
    target_column,
    product_column="Product Name",
    category_column="Category",
    product_name=None,
):
    df = df.copy()

    is_global = (
        product_name is None
        or product_name == "— Global (todos los productos) —"
    )
    is_category = (
        not is_global
        and product_name.startswith(CATEGORY_HEADER_PREFIX)
    )

    if is_global:
        pass  # sin filtro
    elif is_category:
        cat = product_name.replace(CATEGORY_HEADER_PREFIX, "").strip()
        df = df[df[category_column] == cat]
        if df.empty:
            raise ValueError(f"No hay datos para la categoría: '{cat}'")
    else:
        df = df[df[product_column] == product_name]
        if df.empty:
            raise ValueError(f"No hay datos para el producto: '{product_name}'")

    # Parsear fechas
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