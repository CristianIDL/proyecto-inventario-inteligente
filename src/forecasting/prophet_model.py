import os
import pickle
from prophet import Prophet


def train_prophet(df_processed, product_name="global"):
    """
    Entrena un modelo Prophet con el DataFrame procesado (columnas ds, y).
    Guarda el modelo serializado en models/.
    Retorna (model, metrics_dict).
    """
    if df_processed is None or df_processed.empty:
        raise ValueError("No hay datos procesados para entrenar.")

    if len(df_processed) < 2:
        raise ValueError("Se necesitan al menos 2 registros para entrenar Prophet.")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )

    model.fit(df_processed[["ds", "y"]])

    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    safe_name = product_name.replace(" ", "_").replace("/", "-").lower()
    model_path = os.path.join("models", f"prophet_{safe_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Métricas básicas en training set
    future_train = model.make_future_dataframe(periods=0, freq="D")
    forecast_train = model.predict(future_train)

    merged = df_processed.set_index("ds").join(
        forecast_train.set_index("ds")[["yhat"]]
    ).dropna()

    mae = (merged["y"] - merged["yhat"]).abs().mean()
    rmse = ((merged["y"] - merged["yhat"]) ** 2).mean() ** 0.5

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "model_path": model_path,
        "registros": len(df_processed),
        "producto": product_name,
    }

    return model, metrics