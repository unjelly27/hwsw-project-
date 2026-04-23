import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a periocular recognition model with ArcFace-style supervision.")
    parser.add_argument("--data-dir", type=str, default="../periocular_filtered")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "vit_b_16"])
    parser.add_argument("--strategy", type=str, default="arcface_finetune", choices=["arcface_finetune", "linear_probe"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--save-path", type=str, default="periocular_arcface_best.pth")
    parser.add_argument("--history-path", type=str, default="periocular_arcface_history.json")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.02)
        ], p=0.7),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.94, 1.06)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def stratified_split(samples, train_split, seed):
    grouped_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(samples):
        grouped_indices[class_idx].append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for _, indices in grouped_indices.items():
        rng.shuffle(indices)
        split_at = max(1, int(len(indices) * train_split))
        split_at = min(split_at, len(indices) - 1)
        train_indices.extend(indices[:split_at])
        val_indices.extend(indices[split_at:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_sampler(train_targets):
    class_counts = Counter(train_targets)
    sample_weights = [1.0 / class_counts[target] for target in train_targets]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_dataloaders(data_dir, image_size, batch_size, num_workers, train_split, seed):
    train_transform, eval_transform = build_transforms(image_size)

    base_dataset = datasets.ImageFolder(data_dir)
    train_indices, val_indices = stratified_split(base_dataset.samples, train_split, seed)

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    train_eval_dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=eval_transform)

    train_subset = Subset(train_dataset, train_indices)
    train_eval_subset = Subset(train_eval_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    train_targets = [base_dataset.samples[idx][1] for idx in train_indices]
    sampler = build_sampler(train_targets)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_subset, sampler=sampler, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_subset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

    return (
        base_dataset.classes,
        train_loader,
        train_eval_loader,
        val_loader,
        train_targets,
        train_indices,
        val_indices,
    )


def select_device(requested_device):
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available.")

    if requested_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available.")

    if requested_device == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EmbeddingBackbone(nn.Module):
    def __init__(self, arch, embedding_dim, dropout):
        super().__init__()
        self.arch = arch

        if arch == "resnet50":
            backbone = self._load_model(models.resnet50, getattr(models, "ResNet50_Weights").DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        elif arch == "vit_b_16":
            backbone = self._load_model(models.vit_b_16, getattr(models, "ViT_B_16_Weights").DEFAULT)
            in_features = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    @staticmethod
    def _load_model(builder, default_weights):
        try:
            return builder(weights=default_weights)
        except Exception as exc:
            print(f"Could not load pretrained weights, falling back to random init: {exc}")
            return builder(weights=None)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)
        return F.normalize(embedding, dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def cosine_logits(self, embeddings):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine * self.scale

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


def set_backbone_trainability(model, frozen):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = not frozen
    for parameter in model.embedding.parameters():
        parameter.requires_grad = True


def compute_identification_metrics(embeddings, labels):
    if embeddings.size(0) < 2:
        return 0.0, 0.0

    similarity = embeddings @ embeddings.T
    same_identity = labels.unsqueeze(1) == labels.unsqueeze(0)
    diagonal_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

    retrieval_similarity = similarity.masked_fill(diagonal_mask, -1e9)
    nearest_indices = retrieval_similarity.argmax(dim=1)
    rank1 = (labels[nearest_indices] == labels).float().mean().item()

    upper_mask = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    same_scores = similarity[upper_mask & same_identity]
    diff_scores = similarity[upper_mask & (~same_identity)]

    if same_scores.numel() == 0 or diff_scores.numel() == 0:
        verification = 0.0
    else:
        candidate_thresholds = torch.cat([same_scores, diff_scores]).sort().values
        best_verification = 0.0
        for threshold in candidate_thresholds[:: max(1, candidate_thresholds.numel() // 200)]:
            tpr = (same_scores >= threshold).float().mean()
            tnr = (diff_scores < threshold).float().mean()
            best_verification = max(best_verification, ((tpr + tnr) * 0.5).item())
        verification = best_verification

    return rank1, verification


def clone_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def build_class_prototypes(embeddings, labels, num_classes):
    prototypes = torch.zeros(num_classes, embeddings.size(1), dtype=embeddings.dtype)
    counts = torch.zeros(num_classes, dtype=embeddings.dtype)

    prototypes.index_add_(0, labels, embeddings)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=embeddings.dtype))
    counts = counts.clamp_min(1.0).unsqueeze(1)
    prototypes = prototypes / counts
    return F.normalize(prototypes, dim=1)


def compute_gallery_rank1(train_embeddings, train_labels, query_embeddings, query_labels):
    similarity = query_embeddings @ train_embeddings.T
    nearest_indices = similarity.argmax(dim=1)
    predictions = train_labels[nearest_indices]
    return (predictions == query_labels).float().mean().item()


def evaluate(model, margin_head, train_eval_loader, val_loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    train_embeddings = []
    train_labels = []
    val_embeddings = []
    val_labels = []

    with torch.no_grad():
        for images, labels in train_eval_loader:
            images = images.to(device, non_blocking=True)
            embeddings = model(images)
            train_embeddings.append(embeddings.detach().cpu())
            train_labels.append(labels.detach().cpu())

    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    prototypes = build_class_prototypes(train_embeddings, train_labels, num_classes)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            embeddings = model(images)
            logits = margin_head(embeddings, labels)
            prediction_logits = margin_head.cosine_logits(embeddings)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (prediction_logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            val_embeddings.append(embeddings.detach().cpu())
            val_labels.append(labels.detach().cpu())

    val_embeddings = torch.cat(val_embeddings, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    self_rank1, verification = compute_identification_metrics(val_embeddings, val_labels)
    prototype_logits = val_embeddings @ prototypes.T
    prototype_rank1 = (prototype_logits.argmax(dim=1) == val_labels).float().mean().item()
    gallery_rank1 = compute_gallery_rank1(train_embeddings, train_labels, val_embeddings, val_labels)

    return {
        "loss": total_loss / total_samples,
        "classification_acc": total_correct / total_samples,
        "rank1_acc": self_rank1,
        "prototype_rank1_acc": prototype_rank1,
        "gallery_rank1_acc": gallery_rank1,
        "verification_acc": verification,
    }


def save_checkpoint(path, model, margin_head, metadata):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "margin_head_state_dict": margin_head.state_dict(),
        "metadata": metadata,
    }
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)
    print("Using device:", device)

    classes, train_loader, train_eval_loader, val_loader, train_targets, train_indices, val_indices = make_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        seed=args.seed,
    )
    num_classes = len(classes)
    class_counts = Counter(train_targets)

    print(f"Strategy: {args.strategy}")
    print(f"Architecture: {args.arch}")
    print(f"Classes: {num_classes}")
    print(f"Train images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")
    print(f"Min/Max train samples per class: {min(class_counts.values())}/{max(class_counts.values())}")

    model = EmbeddingBackbone(args.arch, args.embedding_dim, args.dropout).to(device)
    margin_head = ArcMarginProduct(
        in_features=args.embedding_dim,
        out_features=num_classes,
        scale=args.scale,
        margin=args.margin,
    ).to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(margin_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    history = []
    best_score = -1.0
    best_state = None
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    effective_freeze_epochs = min(args.freeze_epochs, max(0, args.epochs - 1))

    for epoch in range(args.epochs):
        backbone_frozen = args.strategy == "linear_probe" or epoch < effective_freeze_epochs
        set_backbone_trainability(model, frozen=backbone_frozen)
        model.train()
        margin_head.train()

        running_loss = 0.0
        running_correct = 0
        total_seen = 0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            logits = margin_head(embeddings, labels)
            prediction_logits = margin_head.cosine_logits(embeddings)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (prediction_logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)

            if args.log_interval > 0 and (
                batch_index % args.log_interval == 0 or batch_index == len(train_loader)
            ):
                print(
                    f"Epoch {epoch + 1:02d}/{args.epochs} | "
                    f"Batch {batch_index:03d}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} | "
                    f"running_acc={running_correct / total_seen:.4f} | "
                    f"frozen={backbone_frozen}",
                    flush=True,
                )

        train_loss = running_loss / total_seen
        train_acc = running_correct / total_seen
        metrics = evaluate(model, margin_head, train_eval_loader, val_loader, device, num_classes)
        combined_score = 0.8 * metrics["gallery_rank1_acc"] + 0.2 * metrics["verification_acc"]

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": metrics["loss"],
            "val_classification_acc": metrics["classification_acc"],
            "val_rank1_acc": metrics["rank1_acc"],
            "val_prototype_rank1_acc": metrics["prototype_rank1_acc"],
            "val_gallery_rank1_acc": metrics["gallery_rank1_acc"],
            "val_verification_acc": metrics["verification_acc"],
            "score": combined_score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={metrics['loss']:.4f} cls={metrics['classification_acc']:.4f} "
            f"rank1={metrics['rank1_acc']:.4f} proto={metrics['prototype_rank1_acc']:.4f} "
            f"gallery={metrics['gallery_rank1_acc']:.4f} "
            f"verify={metrics['verification_acc']:.4f} "
            f"| frozen={backbone_frozen}"
        )

        if combined_score > best_score:
            best_score = combined_score
            best_epoch = epoch + 1
            best_metrics = copy.deepcopy(epoch_record)
            epochs_without_improvement = 0
            best_state = {
                "model": clone_state_dict_to_cpu(model.state_dict()),
                "margin_head": clone_state_dict_to_cpu(margin_head.state_dict()),
            }
            save_checkpoint(
                args.save_path,
                model,
                margin_head,
                {
                    "arch": args.arch,
                    "embedding_dim": args.embedding_dim,
                    "image_size": args.image_size,
                    "classes": classes,
                    "best_score": best_score,
                },
            )
            print(f"Saved new best checkpoint to {args.save_path}")
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs without improvement "
                f"(patience={args.early_stop_patience})."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        margin_head.load_state_dict(best_state["margin_head"])

    final_metrics = evaluate(model, margin_head, train_eval_loader, val_loader, device, num_classes)
    Path(args.history_path).write_text(json.dumps(history, indent=2))

    print("\nTraining complete.")
    print(f"Best combined score: {best_score:.4f}")
    if best_metrics is not None:
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation classification accuracy: {best_metrics['val_classification_acc']:.4f}")
        print(f"Best validation Rank-1 accuracy: {best_metrics['val_rank1_acc']:.4f}")
        print(f"Best validation Prototype Rank-1 accuracy: {best_metrics['val_prototype_rank1_acc']:.4f}")
        print(f"Best validation Gallery Rank-1 accuracy: {best_metrics['val_gallery_rank1_acc']:.4f}")
        print(f"Best validation verification accuracy: {best_metrics['val_verification_acc']:.4f}")
    print(f"Reloaded-best classification accuracy: {final_metrics['classification_acc']:.4f}")
    print(f"Reloaded-best Rank-1 accuracy: {final_metrics['rank1_acc']:.4f}")
    print(f"Reloaded-best Prototype Rank-1 accuracy: {final_metrics['prototype_rank1_acc']:.4f}")
    print(f"Reloaded-best Gallery Rank-1 accuracy: {final_metrics['gallery_rank1_acc']:.4f}")
    print(f"Reloaded-best verification accuracy: {final_metrics['verification_acc']:.4f}")
    print(f"Checkpoint saved at: {args.save_path}")
    print(f"Training history saved at: {args.history_path}")


if __name__ == "__main__":
    main()
