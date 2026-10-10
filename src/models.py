from typing import Literal, cast

import torch
from torch import Tensor, nn
from torchvision import models


ModelName = Literal[
    "mobilenet_v2",
    "resnet18",
    "efficientnet_b0"
]


class BiometricEmbeddingModel(nn.Module):
    """
    CNN model for biometric embedding extraction.

    During training:
        image -> backbone -> embedding -> classifier -> identity logits

    During inference:
        image -> backbone -> embedding
    """

    def __init__(self, model_name: ModelName, embedding_dim: int, num_classes: int, pretrained: bool = True):
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.backbone, feature_dim = self._build_backbone(
            model_name=model_name,
            pretrained=pretrained
        )

        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )


    @staticmethod
    def _build_backbone(model_name: ModelName, pretrained: bool) -> tuple[nn.Module, int]:

        if model_name == "mobilenet_v2":
            weights = (
                models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            )

            model = models.mobilenet_v2(
                weights=weights,
            )

            classifier = cast(
                nn.Linear, 
                model.classifier[1]
            )

            feature_dim = classifier.in_features
            backbone = model.features

        elif model_name == "resnet18":
            weights = (
                models.ResNet18_Weights.DEFAULT if pretrained else None
            )

            model = models.resnet18(
                weights=weights,
            )

            classifier = cast(
                nn.Linear, 
                model.fc
            )

            feature_dim = classifier.in_features

            backbone = nn.Sequential(
                *list(model.children())[:-1]
            )

        elif model_name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT
                if pretrained
                else None
            )

            model = models.efficientnet_b0(
                weights=weights
            )

            classifier = cast(
                nn.Linear, 
                model.classifier[1]
            )

            feature_dim = classifier.in_features
            backbone = model.features

        else:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        return backbone, feature_dim


    def forward_features(self, x: Tensor) -> Tensor:
        """
        Extract backbone features before the embedding layer.
        """

        x = self.backbone(x)

        if self.model_name == "mobilenet_v2":
            x = nn.functional.adaptive_avg_pool2d(
                x,
                output_size=1,
            )

        elif self.model_name == "efficientnet_b0":
            x = nn.functional.adaptive_avg_pool2d(
                x,
                output_size=1,
            )

        elif self.model_name == "resnet18":
            # ResNet backbone already contains its final
            # AdaptiveAvgPool2d layer.
            pass

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x


    def forward_embedding(self, x: Tensor) -> Tensor:
        """
        Extract the final biometric embedding.
        """

        features = self.forward_features(x)
        embedding = self.embedding(features)

        return embedding

    def forward(self, x: Tensor) -> Tensor:
        """
        Return biometric-class classification logits.
        """

        embedding = self.forward_embedding(x)
        logits = self.classifier(embedding)

        return logits