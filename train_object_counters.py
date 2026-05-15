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
DEFAULT_CACHE_PATH = "object_count_cache.json"


TARGET_TO_CLASS: Dict[str, str] = {
    "house": "urban_land",
    "tree": "forest_land",
    "road": "urban_land",
}

TARGET_TO_DEFAULT_OUT: Dict[str, str] = {
    "house": "house_counter.pth",
    "tree": "tree_counter.pth",
    "road": "road_counter.pth",
}


class SingleCounterNet(nn.Module):
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
            nn.Linear(128, 1),
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


def get_target_class_name(target: str) -> str:
    return TARGET_TO_CLASS.get(target, target)


def default_out_for_target(target: str) -> str:
    return TARGET_TO_DEFAULT_OUT.get(target, f"{target}_counter.pth")


def profile_for_target(target: str) -> Tuple[float, int, int]:
    # ratio, min_seed, min_obj_pixels
    if target == "house":
        return 0.35, 3, 30
    if target == "tree":
        return 0.25, 2, 20
    if target == "road":
        return 0.20, 2, 40
    return 0.28, 2, 20


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

    ratio, min_seed, min_obj_pixels = profile_for_target(target)
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


def build_common_filenames(image_dir: str, mask_dir: str) -> List[str]:
    image_files = {name for name in os.listdir(image_dir) if name.lower().endswith(".png")}
    mask_files = {name for name in os.listdir(mask_dir) if name.lower().endswith(".png")}
    common = sorted(image_files.intersection(mask_files))
    if not common:
        raise RuntimeError("No common image-mask filenames found.")
    return common


def load_cache(cache_path: str) -> Dict[str, Dict[str, int]]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache: Dict[str, Dict[str, int]], cache_path: str) -> None:
    if not cache_path:
        return
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(cache, file)


class ObjectCountDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        filenames: Sequence[str],
        class_rgb: Tuple[int, int, int],
        target_key: str,
        cache: Dict[str, Dict[str, int]],
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = list(filenames)
        self.class_rgb = class_rgb
        self.target_key = target_key
        self.cache = cache

    def __len__(self) -> int:
        return len(self.filenames)

    def _compute_target(self, filename: str) -> int:
        cache_row = self.cache.setdefault(filename, {})
        if self.target_key in cache_row:
            return int(cache_row[self.target_key])

        mask_path = os.path.join(self.mask_dir, filename)
        mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        binary = rgb_match_mask(mask_rgb, self.class_rgb)
        count = estimate_instances(binary, target=self.target_key)
        cache_row[self.target_key] = int(count)
        return count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        target_count = self._compute_target(filename)
        target = torch.tensor([target_count], dtype=torch.float32)
        return image_tensor, target


def evaluate(model: SingleCounterNet, loader: DataLoader, device: str) -> float:
    model.eval()
    mae_total = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = F.softplus(model(images))
            mae_total += torch.abs(preds[:, 0] - targets[:, 0]).sum().item()
            sample_count += targets.shape[0]
    return mae_total / sample_count if sample_count else 0.0


def train_one_target(
    target_key: str,
    out_path: str,
    class_lookup: ClassLookup,
    common_files: Sequence[str],
    args: argparse.Namespace,
    cache: Dict[str, Dict[str, int]],
    device: str,
) -> None:
    target_class = get_target_class_name(target_key)
    if target_class not in class_lookup.rgb_by_name:
        raise RuntimeError(f"Class '{target_class}' not found in class_dict.csv for target '{target_key}'.")

    rng = random.Random(args.seed)
    files = list(common_files)
    rng.shuffle(files)
    if args.max_samples > 0:
        files = files[: args.max_samples]

    val_count = max(1, int(len(files) * args.val_ratio))
    val_files = files[:val_count]
    train_files = files[val_count:]
    if not train_files:
        raise RuntimeError(f"Training split empty for target '{target_key}'.")

    print(f"\n=== Training target: {target_key} (class={target_class}) ===")
    print(f"Total samples: {len(files)} | Train: {len(train_files)} | Val: {len(val_files)}")

    train_dataset = ObjectCountDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        filenames=train_files,
        class_rgb=class_lookup.rgb_by_name[target_class],
        target_key=target_key,
        cache=cache,
    )
    val_dataset = ObjectCountDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        filenames=val_files,
        class_rgb=class_lookup.rgb_by_name[target_class],
        target_key=target_key,
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

    model = SingleCounterNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best_mae = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                raw = model(images)
                preds = F.softplus(raw)
                loss = F.smooth_l1_loss(torch.log1p(preds), torch.log1p(targets))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batches += 1

        train_loss = running_loss / max(batches, 1)
        val_mae = evaluate(model, val_loader, device=device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_mae={val_mae:.3f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_key": target_key,
                    "target_class": target_class,
                    "epoch": epoch,
                    "val_mae": val_mae,
                    "train_samples": len(train_files),
                    "val_samples": len(val_files),
                    "image_size": 256,
                },
                out_path,
            )
            print(f"Saved best checkpoint -> {out_path}")

    print(f"Done target={target_key}, best_epoch={best_epoch}, best_val_mae={best_mae:.3f}")


def resolve_targets(targets: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for target in targets:
        t = target.strip().lower()
        if not t:
            continue
        if t == "both":
            expanded.extend(["house", "tree", "road"])
        elif t == "all":
            expanded.extend(["house", "tree", "road", "agriculture_land", "rangeland", "water", "barren_land", "unknown"])
        else:
            expanded.append(t)

    seen = set()
    ordered_unique: List[str] = []
    for target in expanded:
        if target not in seen:
            ordered_unique.append(target)
            seen.add(target)
    return ordered_unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train separate object counter models (house/tree/road/labels).")
    parser.add_argument("--image-dir", default=DEFAULT_IMG_DIR, help="Training image folder.")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR, help="Training mask folder.")
    parser.add_argument("--class-dict", default=DEFAULT_CLASS_DICT, help="class_dict.csv path.")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Pseudo-label cache JSON path.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["both"],
        help="Targets to train. Example: house tree road | both | all | water forest_land",
    )
    parser.add_argument("--max-samples", type=int, default=10000, help="Limit number of paired samples (0 = all).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per target.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    targets = resolve_targets(args.targets)
    if not targets:
        raise RuntimeError("No valid targets provided.")

    lookup = load_class_lookup(args.class_dict)
    common_files = build_common_filenames(args.image_dir, args.mask_dir)
    cache = load_cache(args.cache_path)

    for target in targets:
        out_path = default_out_for_target(target)
        train_one_target(
            target_key=target,
            out_path=out_path,
            class_lookup=lookup,
            common_files=common_files,
            args=args,
            cache=cache,
            device=device,
        )

    save_cache(cache, args.cache_path)
    print("All requested target models trained.")


if __name__ == "__main__":
    main()
