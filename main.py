import os
import torch
import pandas as pd
from src.dataset import get_dataloaders, get_transforms
from src.models import get_model, get_target_layer
from src.train import grid_search
from src.evaluate import evaluate_on_test, visualize_gradcam

def main():
    # Setup
    PASTA_IMAGENS = 'images'
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Usando device: {device}")
    
    # Check if images folder exists (mocking it if you are testing without images)
    if not os.path.exists(PASTA_IMAGENS):
        print(f"AVISO: Pasta {PASTA_IMAGENS} não encontrada. O código vai falhar no dataloader.")
        
    df_treino = pd.read_csv('split_treino.csv')
    df_val = pd.read_csv('split_val.csv')
    df_teste = pd.read_csv('split_teste.csv')
    
    print(f"Dados - Treino: {len(df_treino)} | Val: {len(df_val)} | Teste: {len(df_teste)}")
    
    # 1. Configurar Dataloaders
    batch_size = 32
    dl_treino, dl_val, dl_teste = get_dataloaders(df_treino, df_val, df_teste, PASTA_IMAGENS, batch_size=batch_size)
    
    # 2. Configurar Grid Search para vários modelos
    model_names = ['resnet18', 'efficientnet_b0', 'vit_base_patch16_224'] # Pode adicionar 'vit_base_patch16_224'
    params_grid = {
        'lr': [1e-4, 5e-5],
        'weight_decay': [1e-4]
    }
    epocas = 15 # Ajuste conforme necessidade
    
    todas_metricas = []
    
    # Cria pasta de resultados
    os.makedirs('results', exist_ok=True)
    
    for model_name in model_names:
        print(f"\n{'='*50}\nIniciando Pipeline para: {model_name}\n{'='*50}")
        
        # 3. Rodar Grid Search e Treinamento
        best_model_path = grid_search(
            model_name=model_name,
            get_model_fn=get_model,
            dl_treino=dl_treino,
            dl_val=dl_val,
            device=device,
            params_grid=params_grid,
            epocas=epocas
        )
        
        # 4. Avaliar Melhor Modelo no Teste
        print(f"\nCarregando melhor modelo de {best_model_path} para avaliacao no teste...")
        melhor_modelo = get_model(model_name, pretrained=False)
        melhor_modelo.load_state_dict(torch.load(best_model_path, map_location=device))
        melhor_modelo = melhor_modelo.to(device)
        
        metrics = evaluate_on_test(melhor_modelo, dl_teste, device, model_name=model_name)
        todas_metricas.append(metrics)
        
        # 5. GradCAM
        _, transform_avaliacao = get_transforms()
        target_layer = get_target_layer(melhor_modelo, model_name)
        
        if target_layer is not None:
            is_vit = 'vit' in model_name.lower()
            print(f"Gerando GradCAM para {model_name}...")
            visualize_gradcam(melhor_modelo, target_layer, df_teste, PASTA_IMAGENS, transform_avaliacao, device, is_vit=is_vit)
        else:
            print(f"Pulando GradCAM: Camada alvo não encontrada para {model_name}.")
            
    # 6. Salvar todos os resultados em um CSV
    df_results = pd.DataFrame(todas_metricas)
    caminho_csv = 'results/comparativo_modelos.csv'
    df_results.to_csv(caminho_csv, index=False)
    print(f"\n{'='*50}\nPipeline finalizado! Resultados consolidados salvos em: {caminho_csv}\n{'='*50}")
    print(df_results)

if __name__ == "__main__":
    main()

