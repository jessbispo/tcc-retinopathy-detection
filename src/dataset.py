import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 224
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]

def get_transforms():
    transform_treino = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO),
    ])
    transform_avaliacao = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO),
    ])
    return transform_treino, transform_avaliacao

class DatasetRetina(Dataset):
    def __init__(self, dataframe, pasta_imagens, transform):
        self.df = dataframe.reset_index(drop=True)
        self.pasta = pasta_imagens
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, i):
        linha = self.df.iloc[i]
        caminho = os.path.join(self.pasta, linha['file'])
        imagem = Image.open(caminho).convert('RGB')
        imagem = self.transform(imagem)
        # Converter para float para funcionar com MSELoss
        grau = float(linha['final_icdr'])
        return imagem, grau

def get_dataloaders(df_treino, df_val, df_teste, pasta_imagens, batch_size=32, num_workers=0):
    transform_treino, transform_avaliacao = get_transforms()
    
    ds_treino = DatasetRetina(df_treino, pasta_imagens, transform_treino)
    ds_val = DatasetRetina(df_val, pasta_imagens, transform_avaliacao)
    ds_teste = DatasetRetina(df_teste, pasta_imagens, transform_avaliacao)
    
    dl_treino = DataLoader(ds_treino, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_teste = DataLoader(ds_teste, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return dl_treino, dl_val, dl_teste

