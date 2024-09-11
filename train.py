import os

import datasets
import wandb

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from torch.utils.data import DataLoader, default_collate
from torch.cuda.amp import GradScaler, autocast

from models.dataloader import MultiEpochsDataLoader
from models.distill import DistillableViT, DistillableBitViT, DistillWrapper
from models.vit import ViT, BitViT
from models.bitlinear import BitLinear, RMSNorm, FakeQuantLinear, absmax, zeropoint
from models.deepfake import DeepFakeEfficientNet, DeepFakeViT
from models.utils import parameter_stats, estimate_size

import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split, transform):
        self.datasets = datasets.load_from_disk(f"{path}/{split}")
        self.transform = transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        image, label = self.datasets[idx]["image"], self.datasets[idx]["label"]
        return self.transform(image), torch.tensor(label)


def load_train_data(config):
    train_dataset = Dataset(
        config["data"]["path"], "train", config["data"]["preprocess"]["train"]
    )
    val_dataset = Dataset(
        config["data"]["path"], "validation", config["data"]["preprocess"]["validation"]
    )

    collate_fn = default_collate
    if config["data"]["cutmix_or_mixup"]["enabled"]:
        cutmix = transforms_v2.CutMix(
            alpha=config["data"]["cutmix_or_mixup"]["cutmix"],
            num_classes=config["data"]["num_classes"],
        )
        mixup = transforms_v2.MixUp(
            alpha=config["data"]["cutmix_or_mixup"]["mixup"],
            num_classes=config["data"]["num_classes"],
        )
        cutmix_or_mixup = transforms_v2.RandomChoice(
            [cutmix, mixup], p=config["data"]["cutmix_or_mixup"]["p"]
        )

        def custom_collate(batch):
            return cutmix_or_mixup(*default_collate(batch))

        collate_fn = custom_collate

    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        **config["data"]["loader"],
    )

    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        collate_fn=default_collate,
        **config["data"]["loader"],
    )

    return train_loader, val_loader


def train_model(model, optimizer, scheduler, config, start, best):
    train_loader, val_loader = load_train_data(config)
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)
    accum_steps = config["accum_steps"]
    grad_scaler = GradScaler()

    criterion = config["criterion"]["type"](**config["criterion"]["args"])

    for epoch in range(start, config["epochs"]):
        # warmup epochs
        if epoch < config["warmup"]:
            lr = (config["optimizer"]["args"]["lr"] / config["warmup"]) * (epoch + 1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            scheduler.step()

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_length = 0

        batch = 0
        optimizer.zero_grad()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # enable mixed precision training
            with autocast():
                if config["distill"]["enabled"]:
                    loss, outputs = model(inputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            # accumulate gradients
            loss /= accum_steps
            grad_scaler.scale(loss).backward()

            running_loss += loss.item() * inputs.size(0)
            running_acc += torch.sum(
                torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)
            ).item()
            running_length += inputs.size(0)
            batch += 1

            # update weights
            if batch % accum_steps == 0 or batch == len_train_loader:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()

            if (
                config["wandb"]["enabled"]
                and batch % config["wandb"]["log_interval"] == 0
            ):
                wandb.log(
                    {
                        "train/loss": running_loss / running_length,
                        "train/acc": running_acc / running_length,
                        "train/step": batch + epoch * len_train_loader,
                    },
                )

            print(
                f"Epoch {epoch + 1} ({batch} / {len_train_loader}): Train Loss: {running_loss / running_length:.4f}, Train Acc: {running_acc / running_length:.4f}",
                end="\r",
            )

        epoch_loss = running_loss / running_length
        epoch_acc = running_acc / running_length

        print(
            f"\nEpoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}"
        )

        model.eval()

        running_loss = 0.0
        running_acc = 0.0
        running_length = 0
        batch = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if config["distill"]["enabled"]:
                    loss, outputs = model(inputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
                running_length += inputs.size(0)
                batch += 1

                print(
                    f"Epoch {epoch + 1} ({batch} / {len_val_loader}): Val Loss: {running_loss / running_length:.4f}, Val Acc: {running_acc / running_length:.4f}",
                    end="\r",
                )

            val_loss = running_loss / running_length
            val_acc = running_acc / running_length

            print(
                f"\nEpoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if config["wandb"]["enabled"]:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                )

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss,
            }

            # save checkpoints
            torch.save(state, os.path.join(config["path"], "latest_checkpoint.pth"))
            if val_loss < best["val_loss"]:
                print(
                    f"Epoch {epoch + 1}: New best model found! from {best['val_loss']:.4f} to {val_loss:.4f}"
                )
                best["model"] = state["model"]
                best["val_loss"] = val_loss
                torch.save(state, os.path.join(config["path"], "best_checkpoint.pth"))


def test_model(model, config, split="validation"):
    model.eval()
    dataset = Dataset(
        config["data"]["path"], split, config["data"]["preprocess"]["validation"]
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=default_collate,
        **config["data"]["loader"],
    )

    len_loader = len(loader)

    running_acc = 0.0
    running_length = 0
    batch = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            running_acc += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            running_length += inputs.size(0)

            batch += 1
            print(
                f"Test ({batch} / {len_loader}): Test Acc: {running_acc / running_length:.4f}",
                end="\r",
            )

        test_acc = running_acc / running_length
        print(f"\nFinal Test Acc: {test_acc:.4f}")


def load_config(config):
    # create experiment directory
    config["path"] = os.path.join(
        "experiments", config["project"], config["experiment"]
    )
    os.makedirs(config["path"], exist_ok=True)

    # initialize model, optimizer, and scheduler
    model = config["architecture"]["model"](**config["architecture"]["args"]).to(device)
    optimizer = config["optimizer"]["type"](
        model.parameters(), **config["optimizer"]["args"]
    )
    scheduler = config["scheduler"]["type"](optimizer, **config["scheduler"]["args"])

    if config["distill"]["enabled"]:
        distiller = DistillWrapper(student=model, **config["distill"]["args"])
        # distillmixin requires pos_embedding to be on device
        model.pos_embedding = model.pos_embedding.to(device)
        model = distiller.to(device)

    best = {"val_loss": float("inf"), "model": None}
    start = 0

    # load checkpoint if it exists
    checkpoint = os.path.join(config["path"], "latest_checkpoint.pth")
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best["val_loss"] = state["val_loss"]
        start = state["epoch"] + 1
        print(f"checkpoint loaded: {checkpoint}, from epoch {start}")

    return model, optimizer, scheduler, start, best


def train_or_resume(config):
    model, optimizer, scheduler, start, best = load_config(config)

    print(model)
    parameter_stats(model)

    if config["wandb"]["enabled"]:
        wandb.login()
        wandb.init(
            project=config["project"],
            config=config,
            resume="allow",
            id=config["experiment"],
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/acc", step_metric="train/step")
        wandb.define_metric("train/loss", step_metric="train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("val/acc", step_metric="epoch", summary="max")
        wandb.define_metric("val/loss", step_metric="epoch", summary="min")
        wandb.define_metric("lr", step_metric="epoch")

    train_model(
        model,
        optimizer,
        scheduler,
        config,
        start,
        best,
    )

    if config["wandb"]["enabled"]:
        wandb.finish()


def finetune_or_resume(config):
    model, optimizer, scheduler, start, best = load_config(config)

    config["project"] = config["finetune"]["project"]
    config["experiment"] = config["finetune"]["experiment"]
    config["path"] = os.path.join(
        "experiments", config["project"], config["experiment"]
    )
    os.makedirs(config["path"], exist_ok=True)

    deepfake_model = DeepFakeViT(
        model.student, **config["finetune"]["args"]
    )  # assumes distilled model
    model = deepfake_model.to(device)

    print(model)
    stats = parameter_stats(model)

    # reset optimizer and scheduler, and start and best
    optimizer = config["optimizer"]["type"](
        model.parameters(), **config["optimizer"]["args"]
    )
    scheduler = config["scheduler"]["type"](optimizer, **config["scheduler"]["args"])

    best = {"val_loss": float("inf"), "model": None}
    start = 0

    config["distill"]["enabled"] = False

    # check for checkpoint
    checkpoint = os.path.join(config["path"], "latest_checkpoint.pth")
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best["val_loss"] = state["val_loss"]
        start = state["epoch"] + 1
        print(f"finetune checkpoint loaded: {checkpoint}, from epoch {start}")

    # double check desired weights are frozen
    model.freeze()

    if config["wandb"]["enabled"]:
        wandb.login()
        wandb.init(
            project=config["project"],
            config=config,
            resume="allow",
            id=config["experiment"],
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/acc", step_metric="train/step")
        wandb.define_metric("train/loss", step_metric="train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("val/acc", step_metric="epoch", summary="max")
        wandb.define_metric("val/loss", step_metric="epoch", summary="min")
        wandb.define_metric("lr", step_metric="epoch")

    train_model(
        model,
        optimizer,
        scheduler,
        config,
        start,
        best,
    )

    if config["wandb"]["enabled"]:
        wandb.finish()
