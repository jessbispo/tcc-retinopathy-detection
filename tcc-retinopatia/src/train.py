"""
train.py
Script de treino (esqueleto).
Adapte a função `train()` para incluir seu loop de treino e escolha de framework.
"""
import argparse


def train(config_path: str = None):
    """Placeholder da rotina de treino.

    Args:
        config_path: caminho para arquivo de configuração (opcional)
    """
    print(f"Treinamento iniciado. Config: {config_path}")
    # TODO: implementar loop de treino


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo de retinopatia")
    parser.add_argument("--config", type=str, default=None, help="Caminho para arquivo de configuração")
    args = parser.parse_args()
    train(args.config)
