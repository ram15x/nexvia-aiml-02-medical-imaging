"""Classifier registry shared by training and local inference."""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F


ARCHITECTURE_NAMES = {
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "swin_t": "Swin Transformer-Tiny",
    "densenet121_swin_t": "DenseNet-121 + Swin-Tiny feature fusion",
    "densenet121_evidence_concentration_gate": "DenseNet-121 + evidence-concentration gate",
    "densenet121_label_attention": "DenseNet-121 + label-query spatial attention",
    "densenet121_label_graph": "DenseNet-121 + label-query attention + co-occurrence graph",
    "densenet121_label_residual": "DenseNet-121 + residual label-query attention",
    "densenet121_label_graph_residual": "DenseNet-121 + residual label-query attention + co-occurrence graph",
}
IMAGE_SIZE = 224
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def image_to_tensor(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Apply the single versioned RGB/resize/ImageNet normalization path."""
    rgb = image.convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))


class DenseNetSwinFusion(nn.Module):
    """Frozen ImageNet encoders with a trainable multi-label fusion head."""

    def __init__(self, num_labels: int, *, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import (
            DenseNet121_Weights,
            Swin_T_Weights,
            densenet121,
            swin_t,
        )

        self.cnn = densenet121(
            weights=DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.transformer = swin_t(
            weights=Swin_T_Weights.DEFAULT if pretrained else None
        )
        cnn_dim = self.cnn.classifier.in_features
        transformer_dim = self.transformer.head.in_features
        self.cnn.classifier = nn.Identity()
        self.transformer.head = nn.Identity()
        self.fusion_head = nn.Linear(cnn_dim + transformer_dim, num_labels)
        for backbone in (self.cnn, self.transformer):
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
            backbone.eval()

    def train(self, mode: bool = True) -> "DenseNetSwinFusion":
        super().train(mode)
        self.cnn.eval()
        self.transformer.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cnn_features = F.relu(self.cnn.features(images), inplace=False)
        cnn_embedding = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)

        transformer_features = self.transformer.features(images)
        transformer_features = self.transformer.norm(transformer_features)
        transformer_embedding = transformer_features.mean(dim=(1, 2))

        fused = torch.cat((cnn_embedding, transformer_embedding), dim=1)
        return self.fusion_head(fused)


class DenseNetEvidenceConcentrationGate(nn.Module):
    """Blend global and regional evidence using per-label spatial concentration.

    The pretrained DenseNet remains the transfer-learning encoder. The small,
    project-specific head estimates a global score and a regional score for
    each label. It then uses the normalized entropy of that label's evidence
    map to choose how much of the regional score to mix into the global score.
    The bounded residual starts near the global classifier for stable training.
    """

    def __init__(self, num_labels: int, *, pretrained: bool,
                 regional_fraction: float = 0.2) -> None:
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121

        if not 0.0 < regional_fraction <= 1.0:
            raise ValueError("regional_fraction must be in (0, 1]")
        self.encoder = densenet121(
            weights=DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.encoder.classifier = nn.Identity()
        self.global_classifier = nn.Linear(1024, num_labels)
        self.evidence_maps = nn.Conv2d(1024, num_labels, kernel_size=1)
        self.gate_bias = nn.Parameter(torch.full((num_labels,), -2.0))
        self.gate_slope = nn.Parameter(torch.zeros(num_labels))
        self.regional_fraction = regional_fraction

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = F.relu(self.encoder.features(images), inplace=False)
        global_features = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        global_logits = self.global_classifier(global_features)

        evidence = self.evidence_maps(feature_map)
        flat = evidence.flatten(2)
        top_k = max(1, round(flat.shape[-1] * self.regional_fraction))
        regional_logits = flat.topk(top_k, dim=-1).values.mean(dim=-1)

        spatial_log_probs = F.log_softmax(flat, dim=-1)
        spatial_probs = spatial_log_probs.exp()
        entropy = -(spatial_probs * spatial_log_probs).sum(dim=-1)
        concentration = 1.0 - entropy / np.log(flat.shape[-1])
        concentration = concentration.clamp(0.0, 1.0)
        gate = torch.sigmoid(
            self.gate_bias.unsqueeze(0)
            + F.softplus(self.gate_slope).unsqueeze(0) * concentration
        )
        return global_logits + gate * (regional_logits - global_logits)


class DenseNetLabelAttention(nn.Module):
    """DenseNet visual features with optional label-wise graph context."""

    def __init__(self, num_labels: int, *, pretrained: bool,
                 label_adjacency: np.ndarray | torch.Tensor | None = None,
                 use_graph: bool = True,
                 use_global_residual: bool = False,
                 embedding_dim: int = 256) -> None:
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121

        self.encoder = densenet121(
            weights=DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.encoder.classifier = nn.Identity()
        self.global_classifier = (nn.Linear(1024, num_labels)
                                  if use_global_residual else None)
        self.attention_scale = nn.Parameter(torch.zeros(num_labels))
        self.visual_projection = nn.Conv2d(1024, embedding_dim, kernel_size=1)
        self.label_queries = nn.Parameter(torch.empty(num_labels, embedding_dim))
        nn.init.normal_(self.label_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.graph_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.graph_gate = nn.Parameter(torch.tensor(0.0))
        self.use_graph = use_graph
        self.use_global_residual = use_global_residual
        self.classifiers = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 1),
        )

        if label_adjacency is None:
            adjacency = np.eye(num_labels, dtype=np.float32)
        else:
            adjacency = np.asarray(label_adjacency, dtype=np.float32)
            if adjacency.shape != (num_labels, num_labels):
                raise ValueError("label_adjacency must have shape (num_labels, num_labels)")
            if not np.isfinite(adjacency).all() or (adjacency < 0).any():
                raise ValueError("label_adjacency must contain finite non-negative values")
            adjacency = adjacency.copy()
            adjacency = adjacency + np.eye(num_labels, dtype=np.float32)
            row_sum = adjacency.sum(axis=1, keepdims=True)
            adjacency = adjacency / np.maximum(row_sum, 1e-8)
        self.register_buffer("label_adjacency", torch.from_numpy(adjacency))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = F.relu(self.encoder.features(images), inplace=False)
        visual_tokens = self.visual_projection(feature_map).flatten(2).transpose(1, 2)
        queries = self.label_queries.unsqueeze(0).expand(images.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, visual_tokens, visual_tokens,
                                           need_weights=False)
        label_features = self.attention_norm(queries + attended)
        if self.use_graph:
            graph_context = torch.einsum("ij,bjd->bid", self.label_adjacency, label_features)
            label_features = label_features + torch.tanh(self.graph_gate) * self.graph_projection(graph_context)
        attention_logits = self.classifiers(label_features).squeeze(-1)
        if self.use_global_residual:
            pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
            global_logits = self.global_classifier(pooled)
            return global_logits + self.attention_scale.unsqueeze(0) * attention_logits
        return attention_logits


def build_classifier(architecture: str, num_labels: int, *, pretrained: bool,
                     label_adjacency: np.ndarray | torch.Tensor | None = None) -> nn.Module:
    """Build a supported multi-label classifier with a fresh output head."""
    if num_labels < 1:
        raise ValueError("num_labels must be positive")
    if architecture == "densenet121_swin_t":
        return DenseNetSwinFusion(num_labels, pretrained=pretrained)
    if architecture == "densenet121_evidence_concentration_gate":
        return DenseNetEvidenceConcentrationGate(num_labels, pretrained=pretrained)
    if architecture in {
        "densenet121_label_attention", "densenet121_label_graph",
        "densenet121_label_residual", "densenet121_label_graph_residual",
    }:
        return DenseNetLabelAttention(
            num_labels, pretrained=pretrained, label_adjacency=label_adjacency,
            use_graph=architecture in {"densenet121_label_graph", "densenet121_label_graph_residual"},
            use_global_residual=architecture in {"densenet121_label_residual", "densenet121_label_graph_residual"},
        )

    from torchvision.models import (
        DenseNet121_Weights,
        EfficientNet_B0_Weights,
        Swin_T_Weights,
        densenet121,
        efficientnet_b0,
        swin_t,
    )

    if architecture == "densenet121":
        model = densenet121(
            weights=DenseNet121_Weights.DEFAULT if pretrained else None
        )
        model.classifier = nn.Linear(model.classifier.in_features, num_labels)
        return model
    if architecture == "efficientnet_b0":
        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_labels)
        return model
    if architecture == "swin_t":
        model = swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_labels)
        return model
    raise ValueError(f"Unsupported classifier architecture: {architecture}")


def gradcam_target(model: nn.Module, architecture: str) -> tuple[nn.Module, bool]:
    """Return target layer and whether its activation is channels-last."""
    if architecture == "densenet121":
        return model.features, False
    if architecture == "efficientnet_b0":
        return model.features[-1], False
    if architecture == "densenet121_swin_t":
        return model.cnn.features, False
    if architecture == "densenet121_evidence_concentration_gate":
        return model.encoder.features, False
    if architecture in {
        "densenet121_label_attention", "densenet121_label_graph",
        "densenet121_label_residual", "densenet121_label_graph_residual",
    }:
        return model.encoder.features, False
    if architecture == "swin_t":
        return model.features[-1], True
    raise ValueError(f"Grad-CAM is not configured for architecture: {architecture}")
