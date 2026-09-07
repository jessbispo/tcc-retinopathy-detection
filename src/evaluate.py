import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import os
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from .dataset import DESVIO, MEDIA

def evaluate_on_test(modelo, dl_teste, device, model_name="Modelo"):
    modelo.eval()
    preds, reais = [], []
    with torch.no_grad():
        for imagens, graus in dl_teste:
            imagens = imagens.to(device)
            saida = modelo(imagens).squeeze(1)
            
            preds_arredondados = torch.clamp(torch.round(saida), 0, 4)
            preds.extend(preds_arredondados.cpu().numpy())
            reais.extend(graus.numpy())

    preds, reais = np.array(preds), np.array(reais)
    qwk_teste = cohen_kappa_score(reais, preds, weights='quadratic')

    nomes = ['0-Sem RD', '1-Leve', '2-Moderada', '3-Grave', '4-Proliferativa']
    print(f"\nResultados de Teste - {model_name}")
    
    # Generate dictionary report for saving to CSV later
    report_dict = classification_report(reais, preds, target_names=nomes, digits=3, zero_division=0, output_dict=True)
    # Print string report for console
    print(classification_report(reais, preds, target_names=nomes, digits=3, zero_division=0))

    cm = confusion_matrix(reais, preds)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(5), nomes, rotation=45, ha='right')
    plt.yticks(range(5), nomes)
    plt.xlabel('Previsto pelo modelo')
    plt.ylabel('Grau verdadeiro')
    plt.title(f'{model_name} (Teste) - QWK {qwk_teste:.3f}')
    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    
    # Save confusion matrix to file
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/cm_{model_name}.png')
    plt.close() # Close plot so it doesn't block execution in a loop
    
    # Compile metrics
    metrics = {
        'Model': model_name,
        'QWK': qwk_teste,
        'Accuracy': report_dict['accuracy'],
        'Macro_F1': report_dict['macro avg']['f1-score'],
        'Weighted_F1': report_dict['weighted avg']['f1-score']
    }
    return metrics

def desnormalizar(t):
    img = t.numpy().transpose(1, 2, 0)
    img = img * np.array(DESVIO) + np.array(MEDIA)
    return np.clip(img, 0, 1).astype(np.float32)

class RegressionOutputTarget:
    def __init__(self):
        pass
    def __call__(self, model_output):
        return model_output

def visualize_gradcam(modelo, target_layer, df_teste, pasta_imagens, transform_avaliacao, device, is_vit=False):
    """
    GradCAM visualization for regression.
    """
    modelo.eval()
    
    # Função para remodelar o output da ViT de sequências (N, seq, embed) para imagens 2D (N, embed, H, W)
    def reshape_transform(tensor, height=14, width=14):
        # tensor: (batch_size, num_tokens, dim)
        # O primeiro token costuma ser o CLS, ignoramos ele (tensor[:, 1:, :])
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        # Agora permuta para ficar: (batch_size, canais, H, W)
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    if is_vit:
        cam = GradCAM(model=modelo, target_layers=[target_layer], reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=modelo, target_layers=[target_layer])

    graus_alvo = [4, 4, 3, 2, 0]
    selecionadas, cont = [], {}
    for g in graus_alvo:
        linhas = df_teste[df_teste['final_icdr'] == g].reset_index(drop=True)
        if len(linhas) > 0:
            c = cont.get(g, 0)
            if c < len(linhas):
                selecionadas.append(linhas.iloc[c])
                cont[g] = c + 1

    n = len(selecionadas)
    if n == 0:
        print("Nenhuma imagem encontrada para GradCAM.")
        return

    plt.figure(figsize=(13, 5))
    for k, linha in enumerate(selecionadas):
        caminho = os.path.join(pasta_imagens, linha['file'])
        entrada = transform_avaliacao(Image.open(caminho).convert('RGB')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            saida = modelo(entrada).squeeze(1).item()
            previsto = int(max(0, min(4, round(saida))))
            
        real = int(linha['final_icdr'])

        mapa = cam(input_tensor=entrada, targets=[RegressionOutputTarget()])[0]
        base = desnormalizar(entrada[0].cpu())
        sobreposto = show_cam_on_image(base, mapa, use_rgb=True)

        plt.subplot(2, n, k + 1); plt.imshow(base); plt.axis('off')
        plt.title(f"Real: {real} | Previu: {previsto}", fontsize=10)
        plt.subplot(2, n, n + k + 1); plt.imshow(sobreposto); plt.axis('off')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/gradcam_{modelo.__class__.__name__}.png')
    plt.close()
