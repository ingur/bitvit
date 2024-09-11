import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import repeat

from .vit import ViT, BitViT


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class DistillMixin:
    def forward(self, img, distill_token=None):
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, _, _ = x.shape

        x += self.pos_embedding.to(x.device)

        if distilling:
            distill_tokens = repeat(distill_token, "1 1 d -> b 1 d", b=b)
            x = torch.cat((x, distill_tokens), dim=1)

        x = self.transformer(x)

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]

        x = x.mean(dim=1)

        x = self.to_latent(x)
        out = self.mlp_head(x)

        if distilling:
            return out, distill_tokens

        return out


class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs["dim"]
        self.num_classes = kwargs["num_classes"]

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v


class DistillableBitViT(DistillMixin, BitViT):
    def __init__(self, *args, **kwargs):
        super(DistillableBitViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs["dim"]
        self.num_classes = kwargs["num_classes"]

    def to_vit(self):
        v = BitViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v


class DistilledViT(nn.Module):
    def __init__(self, student, distill_token):
        super().__init__()
        assert isinstance(
            student, (DistillableViT, DistillableBitViT)
        ), "student must be either DistillableViT or DistillableBitViT"

        self.__dict__.update(student.__dict__)
        self.distill_token = nn.Parameter(distill_token.detach().clone())

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, _, _ = x.shape

        x += self.pos_embedding

        distill_tokens = repeat(self.distill_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((distill_tokens, x), dim=1)

        x = self.transformer(x)
        x = x[:, :-1]

        x = x.mean(dim=1)

        x = self.to_latent(x)
        out = self.mlp_head(x)

        return out


class DistillWrapper(Module):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature=1.0,
        alpha=0.5,
        hard=False,
        mlp_layernorm=False
    ):
        super().__init__()

        self.teacher = teacher
        self.teacher.eval()

        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim) if mlp_layernorm else nn.Identity(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img, labels, temperature=None, alpha=None, **kwargs):
        alpha = default(alpha, self.alpha)
        T = default(temperature, self.temperature)

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(
            img, distill_token=self.distillation_token, **kwargs
        )
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1).detach(),
                reduction="batchmean",
            )
            distill_loss *= T**2
        else:
            teacher_labels = teacher_logits.argmax(dim=-1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        return loss * (1 - alpha) + distill_loss * alpha, student_logits

    def to_vit(self):
        return DistilledViT(self.student, self.distillation_token)
