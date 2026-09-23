# Execucao local com NVIDIA CUDA

O ambiente virtual do projeto fica em `.venv`. Use seu executavel diretamente;
nao e necessario ativar o ambiente nem alterar a politica de scripts do PowerShell.

Na pasta deste arquivo, verifique o ambiente:

```powershell
.\.venv\Scripts\python.exe run_cuda.py --check
```

Para iniciar uma nova execucao completa:

```powershell
.\.venv\Scripts\python.exe -u run_cuda.py
```

O iniciador imprime GPU, PyTorch, CUDA e device, testa forward/backward na GPU
e somente inicia `main.py` se CUDA funcionar. A verificacao usa um processo
separado, encerrado antes do experimento, preservando o estado aleatorio do treino.
O codigo original do experimento permanece inalterado.

## Ambiente

- Python 3.12.14, reutilizado da instalacao existente em
  `C:\Users\brunn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
  A `.venv` depende dessa instalacao base; mantenha-a no mesmo caminho.
- PyTorch 2.13.0+cu130 e torchvision 0.28.0+cu130, obtidos do indice oficial
  <https://download.pytorch.org/whl/cu130>.
- O CUDA Toolkit nao foi instalado separadamente.
- As versoes completas das dependencias ficam em `logs/environment-freeze.txt`.

O Windows desta maquina tem suporte a caminhos longos desativado. Para instalar
o PyTorch nesta pasta, foi usado o prefixo de caminho estendido `\\?\` no
executavel Python. Isso resolveu o erro WinError 206 sem modificar o registro
do Windows. A execucao normal aceita o caminho convencional mostrado acima.

## Resultados e logs

O programa grava os checkpoints na raiz e os resultados em `results`, como
definido originalmente. Os resultados preexistentes foram copiados para
`bkp/antes_cuda_20260922_231754/results` antes da execucao.

Os logs desta execucao e seus identificadores de processo ficam em `logs`.
Evite iniciar uma segunda execucao enquanto a atual estiver rodando: ambas
usariam os mesmos nomes de checkpoints e resultados.
