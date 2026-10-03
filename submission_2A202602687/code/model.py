"""Required MLP architectures, raw logits and explicit initialization."""
import torch
from torch import nn

EXPECTED_PARAMS = {(256, 128): 47879, (512, 256): 161287, (256, 128, 64): 55687}

class MLP(nn.Module):
    def __init__(self, hidden=(256, 128), dropout=0.0, init="he", in_features=54, num_classes=7):
        super().__init__()
        if not 0 <= dropout < 1: raise ValueError("dropout must be in [0, 1)")
        layers = []
        for width in hidden:
            layers.extend([nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)])
            in_features = width
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(self, x):
        return self.net(x)

def init_weights(model, init):
    if init not in ("zeros", "normal", "xavier", "he", "default"):
        raise ValueError(f"Unknown initialization: {init}")
    if init == "default": return
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if init == "zeros": nn.init.zeros_(layer.weight)
            elif init == "normal": nn.init.normal_(layer.weight, std=0.01)
            elif init == "xavier": nn.init.xavier_normal_(layer.weight)
            elif init == "he": nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def activation_stats(model, x):
    """Standard deviations after hidden ReLUs, followed by final logits."""
    previous = model.training
    model.eval()
    stats = []
    for layer in model.net:
        x = layer(x)
        if isinstance(layer, nn.ReLU): stats.append(float(x.std()))
    stats.append(float(x.std()))
    model.train(previous)
    return stats
