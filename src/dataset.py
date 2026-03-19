"""
dataset.py
Funções e classes para gerenciamento do dataset e carregamento de splits CSV.
"""
from typing import Tuple
import pandas as pd
import os


def load_splits(csv_path: str) -> pd.DataFrame:
    """Carrega CSV de splits (train/val/test) e retorna um DataFrame.

    O CSV deve conter ao menos as colunas: `image_path`, `label`.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    print("Dataset utilities module. Use as library.")
