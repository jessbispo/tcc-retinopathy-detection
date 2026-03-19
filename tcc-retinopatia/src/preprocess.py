"""
preprocess.py
Funções de pré-processamento de imagem (placeholders).

Conteúdo esperado:
- leitura de imagem
- aplicação de CLAHE
- redimensionamento

Este arquivo contém funções exemplo que você pode adaptar.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def preprocess_image(image_path: str, output_path: Optional[str] = None, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Carrega uma imagem, aplica CLAHE e redimensiona.

    Args:
        image_path: caminho do arquivo de entrada
        output_path: caminho para salvar a imagem processada (opcional)
        size: tamanho final (width, height)

    Returns:
        Imagem processada como ndarray (BGR)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # Converter para LAB e aplicar CLAHE no canal L
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Redimensionar
    img_resized = cv2.resize(img_clahe, size, interpolation=cv2.INTER_AREA)

    if output_path:
        cv2.imwrite(output_path, img_resized)

    return img_resized
