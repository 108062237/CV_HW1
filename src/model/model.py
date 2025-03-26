import torchvision.models as models
import torch.nn as nn



model_map = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
    "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
    "resnext50_32x4d": (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.DEFAULT),
    "resnext101_32x8d": (models.resnext101_32x8d, models.ResNeXt101_32X8D_Weights.DEFAULT),
}

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale
    
def build_model(config):
    model_fn, weights = model_map.get(config['model_name'], (None, None))
    if model_fn is None:
        raise ValueError(f"Unknown model name: {config['model_name']}2")
    
    model = model_fn(weights=weights)
    print(f"[Info] Loaded {config['model_name']} with pretrained weights.")

    num_ftrs = model.fc.in_features
    if config.get('use_se', False):
        print("[Info] Applying Squeeze-and-Excitation (SE) module before FC layer.")
        model.avgpool = nn.Sequential(
            model.avgpool,
            SEModule(num_ftrs)
        )
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, config['num_classes'])
        )
    else:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, config['num_classes'])
        )
    return model.to(config['device'])

# testing
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    print(f"model: {config['model_name']}, device: {config['device']}")
    print(f"parameters: {sum([param.numel() for param in model.parameters()])}")

    from torchsummary import summary
    summary(model, input_size=(3, config['image_size'], config['image_size']))
