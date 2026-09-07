# Classificação de Retinopatia Diabética - Refatoração (TCC)

Este repositório contém a implementação do código para o Trabalho de Conclusão de Curso (TCC) envolvendo a classificação de gravidade da Retinopatia Diabética utilizando Deep Learning.

O código original em Jupyter Notebooks foi refatorado para uma estrutura modular em Python, tornando-o mais fácil de ler, manter e testar.

## Principais Alterações
1. **Estrutura Modular:** O código foi dividido em `dataset.py`, `models.py`, `train.py` e `evaluate.py` dentro do diretório `src/`.
2. **Loss Function:** Mudança de `CrossEntropyLoss` (Classificação) para `MSELoss` (Regressão Ordinal), uma vez que a gravidade da doença possui uma ordem intrínseca (0 a 4).
3. **Grid Search:** Implementação de uma busca em grade (Grid Search) no arquivo `train.py` para otimização de hiperparâmetros (Learning Rate, Weight Decay, etc).
4. **Métricas:** As saídas contínuas do modelo são arredondadas para os inteiros mais próximos [0, 4] para o cálculo da Acurácia, Quadratic Weighted Kappa (QWK) e visualização da Matriz de Confusão.

## Estrutura do Projeto

```
.
├── main.py                # Ponto de entrada do programa. Executa Grid Search e Avaliação.
├── src/
│   ├── dataset.py         # Definição do DatasetRetina e transforms (Data Augmentation)
│   ├── models.py          # Instanciação dos modelos (timm) com num_classes=1 (Regressão)
│   ├── train.py           # Loop de treinamento, validação e Grid Search
│   └── evaluate.py        # Geração da Matriz de Confusão, Relatório de Classificação e GradCAM
├── split_treino.csv       # Metadados do conjunto de treino
├── split_val.csv          # Metadados do conjunto de validação
├── split_teste.csv        # Metadados do conjunto de teste
├── images/                # (Requerido) Diretório contendo as imagens
└── README.md              # Este arquivo
```

## Como Executar Localmente

### 1. Pré-requisitos
Certifique-se de que possui o Python instalado (recomendado >= 3.9) e instale as dependências:
```bash
pip install torch torchvision timm pandas numpy matplotlib scikit-learn pillow grad-cam
```

### 2. Organização dos Dados
Certifique-se de que existe uma pasta chamada `images/` na raiz do projeto contendo todas as imagens listadas nos arquivos CSV (ex: `10.1.jpg`).

### 3. Rodando o Treinamento e Avaliação
Basta executar o arquivo principal. Por padrão, ele utilizará a ResNet18 e fará o Grid Search definido em `main.py`:
```bash
python main.py
```
O script fará o seguinte:
1. Carregará os Datasets.
2. Treinará diferentes combinações de hiperparâmetros (Grid Search).
3. Salvará o melhor modelo localmente como `melhor_modelo_resnet18.pth` com base no maior QWK de validação.
4. Carregará o melhor modelo e o avaliará no conjunto de teste.
5. Apresentará a Matriz de Confusão e os heatmaps do GradCAM.

## Testando Novos Modelos / Hiperparâmetros
Para alterar o modelo, modifique a variável `model_name` dentro do `main.py`.
Para testar novos hiperparâmetros na busca, modifique o dicionário `params_grid` no `main.py`.

