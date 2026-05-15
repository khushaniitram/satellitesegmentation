# Model training script for house and tree counter
import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_IMG_DIR = os.path.join("dataset", "train_images_512")
DEFAULT_MASK_DIR = os.path.join("dataset", "train_masks_512")
DEFAULT_CLASS_DICT = "class_dict.csv"
DEFAULT_OUT_PATH = "house_tree_counter.pth"
DEFAULT_CACHE_PATH = "house_tree_count_cache.json"


class CounterNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@dataclass
class ClassLookup:
    rgb_by_name: Dict[str, Tuple[int, int, int]]


def load_class_lookup(class_dict_path: str) -> ClassLookup:
    rgb_by_name: Dict[str, Tuple[int, int, int]] = {}
    with open(class_dict_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rgb_by_name[row["name"].strip()] = (int(row["r"]), int(row["g"]), int(row["b"]))
    return ClassLookup(rgb_by_name=rgb_by_name)


def estimate_instances(binary: np.ndarray, target: str) -> int:
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    if np.count_nonzero(cleaned) == 0:
        return 0

    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    if max_dist <= 0.0:
        return 0

    ratio = 0.35 if target == "house" else 0.25
    min_seed = 3 if target == "house" else 2
    min_obj_pixels = 30 if target == "house" else 20

    seeds = (dist > ratio * max_dist).astype(np.uint8)
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)

    count = 0
    for label_idx in range(1, num_labels):
        seed_area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if seed_area >= min_seed:
            count += 1

    if count == 0 and int(np.count_nonzero(cleaned)) >= min_obj_pixels:
        count = 1
    return count


def rgb_match_mask(mask_rgb: np.ndarray, rgb: Tuple[int, int, int]) -> np.ndarray:
    return np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=2).astype(np.uint8)


class HouseTreeCountDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        filenames: Sequence[str],
        house_rgb: Tuple[int, int, int],
        tree_rgb: Tuple[int, int, int],
        cache: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = list(filenames)
        self.house_rgb = house_rgb
        self.tree_rgb = tree_rgb
        self.cache = cache if cache is not None else {}

    def __len__(self) -> int:
        return len(self.filenames)

    def _compute_target(self, filename: str) -> Tuple[int, int]:
        cached = self.cache.get(filename)
        if cached is not None:
            return int(cached["house"]), int(cached["tree"])

        mask_path = os.path.join(self.mask_dir, filename)
        mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        house_mask = rgb_match_mask(mask_rgb, self.house_rgb)
        tree_mask = rgb_match_mask(mask_rgb, self.tree_rgb)

        house_count = estimate_instances(house_mask, target="house")
        tree_count = estimate_instances(tree_mask, target="tree")
        self.cache[filename] = {"house": int(house_count), "tree": int(tree_count)}
        return house_count, tree_count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        house_count, tree_count = self._compute_target(filename)
        target = torch.tensor([house_count, tree_count], dtype=torch.float32)
        return image_tensor, target


def build_common_filenames(image_dir: str, mask_dir: str) -> List[str]:
    image_files = {name for name in os.listdir(image_dir) if name.lower().endswith(".png")}
    mask_files = {name for name in os.listdir(mask_dir) if name.lower().endswith(".png")}
    common = sorted(image_files.intersection(mask_files))
    if not common:
        raise RuntimeError("No common image-mask filenames found.")
    return common


def load_cache(cache_path: Optional[str]) -> Dict[str, Dict[str, int]]:
    if not cache_path:
        return {}
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache: Dict[str, Dict[str, int]], cache_path: Optional[str]) -> None:
    if not cache_path:
        return
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(cache, file)


def evaluate(model: CounterNet, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    mae_house_total = 0.0
    mae_tree_total = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = F.softplus(model(images))

            mae_house_total += torch.abs(preds[:, 0] - targets[:, 0]).sum().item()
            mae_tree_total += torch.abs(preds[:, 1] - targets[:, 1]).sum().item()
            sample_count += targets.shape[0]

    if sample_count == 0:
        return 0.0, 0.0
    return mae_house_total / sample_count, mae_tree_total / sample_count


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    lookup = load_class_lookup(args.class_dict)
    if "urban_land" not in lookup.rgb_by_name or "forest_land" not in lookup.rgb_by_name:
        raise RuntimeError("class_dict.csv must contain 'urban_land' and 'forest_land'.")

    common_files = build_common_filenames(args.image_dir, args.mask_dir)
    rng = random.Random(args.seed)
    rng.shuffle(common_files)

    if args.max_samples > 0:
        common_files = common_files[: args.max_samples]

    val_count = max(1, int(len(common_files) * args.val_ratio))
    val_files = common_files[:val_count]
    train_files = common_files[val_count:]
    if not train_files:
        raise RuntimeError("Training split is empty. Increase max_samples or reduce val_ratio.")

    print(f"Total samples: {len(common_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

    cache = load_cache(args.cache_path)
    train_dataset = HouseTreeCountDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        filenames=train_files,
        house_rgb=lookup.rgb_by_name["urban_land"],
        tree_rgb=lookup.rgb_by_name["forest_land"],
        cache=cache,
    )
    val_dataset = HouseTreeCountDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        filenames=val_files,
        house_rgb=lookup.rgb_by_name["urban_land"],
        tree_rgb=lookup.rgb_by_name["forest_land"],
        cache=cache,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = CounterNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=device == "cuda")

    best_score = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                raw = model(images)
                preds = F.softplus(raw)
                loss = F.smooth_l1_loss(torch.log1p(preds), torch.log1p(targets))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batches += 1

        train_loss = running_loss / max(batches, 1)
        val_mae_house, val_mae_tree = evaluate(model, val_loader, device=device)
        score = val_mae_house + val_mae_tree
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_mae_house={val_mae_house:.3f} | val_mae_tree={val_mae_tree:.3f}"
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mae_house": val_mae_house,
                    "val_mae_tree": val_mae_tree,
                    "train_samples": len(train_files),
                    "val_samples": len(val_files),
                    "image_size": 256,
                },
                args.out_path,
            )
            print(f"Saved best checkpoint -> {args.out_path}")

    save_cache(cache, args.cache_path)
    print(f"Done. Best epoch: {best_epoch}, best score: {best_score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train house/tree count regressor from segmentation dataset.")
    parser.add_argument("--image-dir", default=DEFAULT_IMG_DIR, help="Training image folder.")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR, help="Training mask folder.")
    parser.add_argument("--class-dict", default=DEFAULT_CLASS_DICT, help="class_dict.csv path.")
    parser.add_argument("--out-path", default=DEFAULT_OUT_PATH, help="Output model checkpoint path.")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Pseudo-label cache JSON path.")
    parser.add_argument("--max-samples", type=int, default=6000, help="Limit number of paired samples (0 = all).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
