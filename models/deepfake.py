import torch
from torch import nn
from .vit import ViT, BitViT

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class DeepFakeClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout):
        super().__init__()
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepFakeEfficientNet(nn.Module):
    def __init__(self, unfreeze_blocks=6, hidden_sizes=[512, 256, 128], dropout=0.2):
        super().__init__()
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        input_features = self.model.classifier[1].in_features

        for param in self.model.parameters():
            param.requires_grad = False

        # always unfreeze the final Conv2dNormActivation layer (stage 7)
        for param in self.model.features[7].parameters():
            param.requires_grad = True

        # unfreeze the specified number of MBConv blocks from the end of stage 6
        total_blocks = len(self.model.features[6])
        for i in range(min(unfreeze_blocks, total_blocks)):
            for param in self.model.features[6][-(i + 1)].parameters():
                param.requires_grad = True

        # unfreeze all BatchNorm layers
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

        self.model.classifier = DeepFakeClassifier(
            input_features, hidden_sizes, dropout
        )

    def forward(self, x):
        return self.model(x)


class DeepFakeViT(nn.Module):
    def __init__(
        self, model, unfreeze_blocks=6, hidden_sizes=[512, 256, 128], dropout=0.2
    ):
        super().__init__()
        self.model = model
        self.unfreeze_blocks = unfreeze_blocks

        input_features = self.model.mlp_head.in_features

        self.model.mlp_head = DeepFakeClassifier(input_features, hidden_sizes, dropout)
        self.freeze()

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze transformer blocks
        blocks = len(self.model.transformer.layers) - 1
        for i in range(self.unfreeze_blocks):
            for param in self.model.transformer.layers[blocks - i].parameters():
                param.requires_grad = True

        # unfreeze mlp_head
        for param in self.model.mlp_head.parameters():
            param.requires_grad = True
