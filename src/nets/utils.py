import torch
from torch import nn
from typing import List


ACTIVATION_MAPPING = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "prelu": nn.PReLU(),
    "celu": nn.CELU()
}
NORMALIZER_MAPPING = {
    "dropout": nn.Dropout,
    "batch_norm": nn.BatchNorm1d,
    "layer_norm": nn.LayerNorm,
    "instance_norm": nn.InstanceNorm1d,
    "batch_norm_2d": nn.BatchNorm2d,
    "instance_nrom_2d": nn.InstanceNorm2d,
}


def normalizer_layer(normalizers, input_dim, prob=0.4):
    norm_layers = [nn.Identity()]
    for norm, norm_str in normalizers:
        if norm_str == "dropout":
            norm_layers.append(norm(prob, inplace=True))
        else:
            norm_layers.append(norm(input_dim))
    return nn.Sequential(*norm_layers)


def init_weights(weights, name, gain: float = 2 ** 0.5, scale : float = 3e-3):
    for p in weights.parameters():
        if name == "normal":
            p.data.normal_(scale)
        elif name == "uniform":
            p.data.uniform_(-scale, scale)
        elif name == "xavier":
            nn.init.xavier_uniform_(p.data)
        elif name == "orthogonal":
            nn.init.orthogonal_(p.data, gain=gain)
        else:
            p.data.zero_()


def create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    weight_init: dict,
    activation: str = "relu",
    normalizer_list: List[dict] = [],
    normalizer_kwargs: dict = {},
):
    activation = nn.Identity() if activation is None else ACTIVATION_MAPPING[activation]
    normalize = [(NORMALIZER_MAPPING[norm], norm) for norm in normalizer_list]

    layers = []
    current_dim = input_dim
    for layer_dim in hidden_dims:
        weights = nn.Linear(current_dim, layer_dim)
        normalizer_kwargs["input_dim"] = layer_dim
        layers.append(weights)
        layers.append(normalizer_layer(normalizers=normalize, **normalizer_kwargs))
        layers.append(activation)
        init_weights(weights, **weight_init)
        current_dim = layer_dim
    return nn.Sequential(*layers)


def create_gauss_filter(size, device, std=None):
    std = size * 0.5 if std is None else std
    n = torch.arange(0, size) - ( size - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    filter_ = torch.outer(w, w)
    filter_ = filter_ / filter_.sum()
    gauss_filter = nn.Conv2d(
        1, 1, size, padding=size // 2, device=device, bias=False, padding_mode="replicate"
    )
    gauss_filter.weight = nn.Parameter(
        filter_.reshape(gauss_filter.weight.shape).to(device), requires_grad=False
    )
    return gauss_filter
