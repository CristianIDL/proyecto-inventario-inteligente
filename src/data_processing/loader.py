import pandas as pd


def load_csv(filepath):
    """
    Carga un archivo CSV y devuelve un DataFrame.
    
    """

    return pd.read_csv(filepath)