import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        if self.elementwise_affine:
            return self.scale * x_norm
        return x_norm


def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (x * scale).round().clamp_(-128, 127) / scale


def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * scale).round().clamp_(-1, 1) / scale


class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x):
        x_quant = x + (activation_quant(x) - x).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x_quant, w_quant, self.bias)


def absmax(x, min_val, max_val, dim=-1):
    scale = max_val / x.abs().max(dim=dim, keepdim=True).values.clamp_(min=1e-5)
    return (x * scale).round().clamp(min_val, max_val) / scale


def zeropoint(x, min_val, max_val, dim=-1):
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    scale = (2 * max_val) / (x_max - x_min)
    zero_point = (min_val - x_min * scale).round()
    x_quant = (x * scale + zero_point).round().clamp(min_val, max_val)
    return (x_quant - zero_point) / scale


class FakeQuantLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        weight_quant=None,
        weight_prec=8,
        act_quant=None,
        act_prec=8,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight_quant = weight_quant  # "absmax", "zeropoint" or None
        self.act_quant = act_quant  # "absmax", "zeropoint" or None
        self.max_weight = 2 ** (weight_prec - 1) - 1
        self.min_weight = -self.max_weight
        self.max_act = 2 ** (act_prec - 1) - 1
        self.min_act = -self.max_act

        self.wq = zeropoint if self.weight_quant == "zeropoint" else absmax
        self.aq = zeropoint if self.act_quant == "zeropoint" else absmax

    def forward(self, x):
        w = self.weight
        if self.act_quant:
            x = self.aq(x, self.min_act, self.max_act, -1)
        if self.weight_quant:
            w = self.wq(self.weight, self.min_weight, self.max_weight, 0)
        return F.linear(x, w, self.bias)
