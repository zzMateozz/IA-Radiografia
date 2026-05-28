"""
Modelo DenseNet-121 estilo CheXNet para clasificacion multi-etiqueta de
patologias toracicas en NIH ChestX-ray14.

Referencia:
    Rajpurkar, P. et al. (2017) "CheXNet: Radiologist-Level Pneumonia
    Detection on Chest X-Rays with Deep Learning", arXiv:1711.05225.

Decisiones (justificadas en docs/01_fase_analisis.md):

1. ARQUITECTURA: DenseNet-121 con pesos pre-entrenados en ImageNet.
   - Las conexiones densas reutilizan caracteristicas a multiples escalas,
     util para capturar patologias correlacionadas (seccion 4.2 del EDA).
   - 121 capas son suficientes y mas eficientes que ResNet-50 / EfficientNet-B4
     para un primer modelo en un equipo limitado.
   - Es la arquitectura de referencia en la literatura de NIH-CXR.

2. CABEZA DE CLASIFICACION: la capa final original (1000 clases ImageNet) se
   reemplaza por una Linear(1024 -> 14). NO se aplica softmax/sigmoid en
   forward(): los LOGITS se devuelven crudos para que la perdida
   BCEWithLogitsLoss los procese (es numericamente mas estable).

3. INPUT: tensor de shape (B, 3, 224, 224) con normalizacion ImageNet.
   Las radiografias monocromaticas se replican a 3 canales en transforms.py.

4. INICIALIZACION: la cabeza nueva se inicializa con Xavier uniforme.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


N_CLASSES = 14


class DenseNetCheXNet(nn.Module):
    """
    DenseNet-121 con cabeza multi-etiqueta de 14 salidas.

    Args:
        n_classes: numero de clases de salida (default 14).
        pretrained: si carga pesos pre-entrenados en ImageNet.
        dropout: probabilidad de dropout antes de la capa final (0.0 desactiva).
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        in_features = backbone.classifier.in_features
        if dropout > 0.0:
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, n_classes),
            )
            head_linear = backbone.classifier[1]
        else:
            backbone.classifier = nn.Linear(in_features, n_classes)
            head_linear = backbone.classifier

        nn.init.xavier_uniform_(head_linear.weight)
        nn.init.zeros_(head_linear.bias)

        self.backbone = backbone
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits de shape (B, n_classes). Aplica sigmoid solo en inferencia."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Conveniencia para inferencia: aplica sigmoid sobre los logits."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    @property
    def feature_extractor(self) -> nn.Module:
        """
        Devuelve el bloque convolucional (sin la capa final).
        Util para Grad-CAM, que necesita acceder al ultimo mapa de activaciones
        antes del global average pooling.
        """
        return self.backbone.features


def build_model(
    n_classes: int = N_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> DenseNetCheXNet:
    """Helper para instanciar el modelo desde scripts/notebooks."""
    return DenseNetCheXNet(n_classes=n_classes, pretrained=pretrained, dropout=dropout)


def count_parameters(model: nn.Module) -> dict:
    """Cuenta parametros entrenables y totales (para reportes)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


if __name__ == "__main__":
    model = build_model(pretrained=False)
    info = count_parameters(model)
    print(f"DenseNetCheXNet -> {info['total']:,} parametros totales "
          f"({info['trainable']:,} entrenables)")

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Input  shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)} (logits, antes de sigmoid)")
