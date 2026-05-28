"""
Pipeline de preprocesamiento y data augmentation para radiografias de torax.

Decisiones (justificadas en docs/01_fase_analisis.md, seccion 8):

1. Las imagenes originales tienen resolucion media de 2,646 x 2,486 px.
   Se redimensionan a 224 x 224 para usar modelos pre-entrenados en ImageNet
   (DenseNet-121, ResNet-50). Es el tamano estandar de CheXNet.

2. Las radiografias son monocromaticas (1 canal). Los modelos ImageNet esperan
   3 canales RGB. Se replica el canal de gris 3 veces.

3. Normalizacion con la media y desviacion estandar de ImageNet:
       mean = [0.485, 0.456, 0.406]
       std  = [0.229, 0.224, 0.225]
   Es el estandar para transfer learning desde modelos ImageNet.

4. Data augmentation SOLO en train. Conservadora porque la anatomia toracica
   tiene una orientacion fija (corazon a la izquierda, arco aortico a la derecha):
       - Rotacion suave (+/- 7 grados).
       - Traslaciones suaves (+/- 5%).
       - Cambio leve de brillo y contraste.
       - NO se usa horizontal flip (cambiaria la lateralidad cardiaca).

5. En val/test SOLO se aplica resize y normalizacion (sin augmentation).
"""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T


IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = IMAGE_SIZE) -> Callable:
    """
    Pipeline para entrenamiento. Incluye data augmentation.

    El input esperado es un PIL.Image en modo 'L' (grayscale) o 'RGB'.
    """
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((image_size, image_size), antialias=True),
        T.RandomAffine(
            degrees=7,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> Callable:
    """Pipeline para validacion y test. SIN data augmentation."""
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((image_size, image_size), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transforms(split: str, image_size: int = IMAGE_SIZE) -> Callable:
    """
    Helper que devuelve la transformacion correspondiente al split.

    Args:
        split: 'train' aplica augmentation, cualquier otro valor solo eval.
        image_size: lado del cuadrado de salida en pixeles.
    """
    if split == "train":
        return get_train_transforms(image_size)
    return get_eval_transforms(image_size)
