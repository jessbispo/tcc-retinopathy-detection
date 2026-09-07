import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, accuracy_score

def evaluate_model(model, loader, device, criterion):
    model.eval()
    preds_todos, reais_todos, perda_total = [], [], 0
    with torch.no_grad():
        for imagens, graus in loader:
            imagens, graus = imagens.to(device), graus.to(device, dtype=torch.float32)
            saida = model(imagens).squeeze(1) # shape (batch,)
            perda = criterion(saida, graus)
            perda_total += perda.item()
            
            # For metrics, round the regression output to nearest integer and clamp between 0 and 4
            preds_arredondados = torch.clamp(torch.round(saida), 0, 4)
            
            preds_todos.extend(preds_arredondados.cpu().numpy())
            reais_todos.extend(graus.cpu().numpy())
            
    qwk = cohen_kappa_score(reais_todos, preds_todos, weights='quadratic')
    acc = accuracy_score(reais_todos, preds_todos)
    return perda_total/len(loader), qwk, acc

def train_one_epoch(model, loader, device, otimizador, criterion):
    model.train()
    perda_treino = 0
    for imagens, graus in loader:
        imagens, graus = imagens.to(device), graus.to(device, dtype=torch.float32)
        otimizador.zero_grad()
        saida = model(imagens).squeeze(1)
        perda = criterion(saida, graus)
        perda.backward()
        otimizador.step()
        perda_treino += perda.item()
    return perda_treino / len(loader)

def grid_search(model_name, get_model_fn, dl_treino, dl_val, device, params_grid, epocas=10):
    """
    Executes a simple grid search over parameters and returns the best model path.
    params_grid: dict of lists, e.g. {'lr': [1e-3, 1e-4], 'weight_decay': [0, 1e-4]}
    """
    from itertools import product
    
    keys, values = zip(*params_grid.items())
    combinacoes = [dict(zip(keys, v)) for v in product(*values)]
    
    melhor_qwk_global = -1
    melhor_params = None
    caminho_melhor_modelo = f'melhor_modelo_{model_name}.pth'
    
    print(f"Iniciando Grid Search para {model_name} com {len(combinacoes)} combinacoes...")
    
    for i, params in enumerate(combinacoes):
        print(f"\n[{i+1}/{len(combinacoes)}] Testando parametros: {params}")
        
        modelo = get_model_fn(model_name).to(device)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=params['lr'], weight_decay=params.get('weight_decay', 0))
        criterio = nn.MSELoss(reduction='mean')
        
        melhor_qwk_local = -1
        
        for epoca in range(1, epocas + 1):
            perda_treino = train_one_epoch(modelo, dl_treino, device, otimizador, criterio)
            perda_val, qwk_val, acc_val = evaluate_model(modelo, dl_val, device, criterio)
            
            print(f"  Epoca {epoca:2d} | Perda Treino: {perda_treino:.3f} | Perda Val: {perda_val:.3f} | QWK Val: {qwk_val:.3f} | Acc Val: {acc_val:.3f}")
            
            if qwk_val > melhor_qwk_local:
                melhor_qwk_local = qwk_val
                
                if qwk_val > melhor_qwk_global:
                    melhor_qwk_global = qwk_val
                    melhor_params = params
                    torch.save(modelo.state_dict(), caminho_melhor_modelo)
                    print(f"  --> Novo melhor modelo global salvo! QWK: {melhor_qwk_global:.3f}")
                    
    print(f"\nGrid Search Concluido!")
    print(f"Melhor QWK de Validacao: {melhor_qwk_global:.3f}")
    print(f"Melhores parametros: {melhor_params}")
    return caminho_melhor_modelo

