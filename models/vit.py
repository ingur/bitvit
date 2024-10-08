import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from .bitlinear import BitLinear, RMSNorm


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels=3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels
        self.to_patch_tokens = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, linear=nn.Linear):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            RMSNorm(hidden_dim),
            linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, linear=nn.Linear):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = RMSNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = linear(dim, inner_dim * 3, bias=False)
        self.to_out = linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, linear=nn.Linear):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            linear=linear,
                        ),
                        FeedForward(dim, mlp_dim, linear=linear),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        spt=False,
        sincos2d=False,
        linear=nn.Linear
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # NOTE: We used full precision for patch embedding, pos_embedding, and mlp_head

        if spt:
            self.to_patch_embedding = SPT(
                dim=dim, patch_size=patch_size, channels=channels
            )
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_height,
                    p2=patch_width,
                ),
                RMSNorm(patch_dim),
                nn.Linear(patch_dim, dim),
            )

        if sincos2d:
            # sincos2d positional embedding (not learned)
            self.pos_embedding = posemb_sincos_2d(
                h=image_height // patch_height,
                w=image_width // patch_width,
                dim=dim,
            )
        else:
            # learned positional embedding
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, linear)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(img.device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.mlp_head(x)


class BitViT(ViT):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        spt=False,
        sincos2d=False,
        linear=BitLinear
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=dim_head,
            spt=spt,
            sincos2d=sincos2d,
            linear=linear,
        )
