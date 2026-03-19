# TCC - Detecção de Retinopatia Diabética (estrutura)

Este repositório contém a estrutura inicial para o trabalho de TCC sobre detecção de retinopatia diabética. Está organizada para facilitar o pré-processamento, treinamento e avaliação de modelos de visão computacional.

Visão geral da estrutura

tcc-retinopatia/
├── data/
│   ├── raw/          ← imagens originais do Messidor-2 (ou outra fonte)
│   ├── processed/    ← imagens após CLAHE e resize
│   └── splits/       ← CSVs com splits de treino/val/teste
├── notebooks/        ← notebooks para exploração e análise
├── src/
│   ├── preprocess.py ← funções de pré-processamento (CLAHE, resize)
│   ├── dataset.py    ← utilitários de dataset e carregamento de CSVs
│   ├── train.py      ← esqueleto do script de treino
│   └── evaluate.py   ← esqueleto de avaliação e métricas
├── models/           ← pesos salvos dos modelos
├── results/          ← métricas, gráficos, matriz de confusão
└── requirements.txt  ← dependências básicas

Como usar (rápido)

1) Preparar ambiente (Windows PowerShell):

```powershell
# criar e ativar virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# instalar dependências
pip install -r .\tcc-retinopatia\requirements.txt
```

2) Adicionar dados
- Coloque as imagens originais em `tcc-retinopatia/data/raw/`.
- Gere as imagens processadas com CLAHE e redimensionamento em `tcc-retinopatia/data/processed/` (o `src/preprocess.py` já contém uma função de exemplo `preprocess_image`).
- Crie CSVs de splits (treino/val/test) em `tcc-retinopatia/data/splits/`. Cada CSV deve ter pelo menos as colunas `image_path` e `label`.

3) Executar scripts (esqueleto)
- Pré-processamento: personalize e rode `src/preprocess.py` para processar imagens em lote.
- Treino: execute `python src/train.py --config caminho_para_config` (o arquivo é um esqueleto; adapte para o framework que escolher — PyTorch ou TensorFlow).
- Avaliação: use `src/evaluate.py` para calcular métricas e gerar gráficos.

Dicas rápidas
- O projeto usa CLAHE (ex.: no canal L do espaço LAB) para melhorar contraste em imagens de retina; isso já está no `preprocess.py` como exemplo.
- Mantenha cópias originais em `data/raw/` e processe para `data/processed/` para reprodutibilidade.
- Salve pesos treinados em `models/` e resultados (plots, CSVs de métricas) em `results/`.

Próximos passos sugeridos
- Implementar processamento em lote (iterar sobre `data/raw/` e gravar em `data/processed/`).
- Implementar um Dataset e DataLoader específicos (ex.: `torch.utils.data.Dataset`) em `src/dataset.py`.
- Implementar loop de treino (checkpoint, early stopping) em `src/train.py`.
- Criar notebooks em `notebooks/` para análise exploratória e visualização (matrizes de confusão, ROC, exemplos corretos/errados).

Referências
- Messidor-2: conjunto de imagens frequentemente usado em estudos de retinopatia diabética.

Licença
- Adicione aqui a licença que for utilizar (ex.: MIT) se desejar.

Contato
- Coloque informações de contato ou orientador aqui, se quiser.
