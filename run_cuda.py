"""Verifica o ambiente CUDA e executa o main.py original em outro processo."""

import argparse
from pathlib import Path
import subprocess
import sys


def check_cuda():
    import torch

    available = torch.cuda.is_available()
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Executavel: {sys.executable}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA do PyTorch: {torch.version.cuda}", flush=True)
    print(f"torch.cuda.is_available(): {available}", flush=True)
    if not available:
        raise RuntimeError("CUDA indisponivel. O experimento nao sera iniciado.")

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("Device do experimento: cuda (cuda:0)", flush=True)
    # Verifica kernels CUDA e cuDNN, incluindo backward, antes do experimento.
    layer = torch.nn.Conv2d(3, 4, kernel_size=3).to("cuda")
    inputs = torch.ones(2, 3, 16, 16, device="cuda", requires_grad=True)
    output = layer(inputs)
    output.square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("O teste CUDA produziu valores nao finitos.")
    if inputs.grad is None or not torch.isfinite(inputs.grad).all().item():
        raise RuntimeError("O teste de gradientes CUDA falhou.")
    print("Teste CUDA/cuDNN (forward e backward): OK", flush=True)
    del inputs, output, layer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Somente verificar CUDA.")
    args = parser.parse_args()
    if args.check:
        check_cuda()
        return 0

    # O teste termina antes do treino, liberando a GPU e preservando o estado aleatorio.
    root = Path(__file__).resolve().parent
    probe = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--check"], cwd=root)
    if probe.returncode != 0:
        return probe.returncode
    print("Iniciando main.py com as configuracoes originais...", flush=True)
    result = subprocess.run([sys.executable, "-u", str(root / "main.py")], cwd=root)
    print(f"main.py encerrado com codigo {result.returncode}.", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
