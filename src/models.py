import timm

def get_model(model_name='resnet18', pretrained=True):
    """
    Instantiates a model for regression (num_classes=1).
    Supported models: resnet18, efficientnet_b0, mobilenetv3_large_100, vit_base_patch16_224
    """
    # Use num_classes=1 for regression (MSELoss)
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
    return model

def get_target_layer(model, model_name):
    """
    Returns the appropriate target layer for GradCAM depending on the architecture.
    """
    model_name = model_name.lower()
    if 'resnet' in model_name:
        return model.layer4[-1]
    elif 'vit' in model_name:
        return model.blocks[-1].norm1
    elif 'efficientnet' in model_name:
        return model.conv_head
    elif 'mobilenet' in model_name:
        return model.blocks[-1]
    else:
        print(f"Aviso: Camada alvo para GradCAM não mapeada para {model_name}.")
        return None
