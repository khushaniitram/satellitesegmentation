import argparse
import csv
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MODEL_PATH = "multiclass_unet.pth"
DEFAULT_CLASS_DICT_PATH = "class_dict.csv"
DEFAULT_IMAGE_DIR = os.path.join("dataset", "train_images_512")
DEFAULT_DETECTIONS_DIR = "detections"
DEFAULT_COUNTER_MODEL_PATH = "house_tree_counter.pth"
DEFAULT_HOUSE_COUNTER_MODEL_PATH = "house_counter.pth"
DEFAULT_TREE_COUNTER_MODEL_PATH = "tree_counter.pth"
DEFAULT_ROAD_COUNTER_MODEL_PATH = "road_counter.pth"


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 3, features: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        bottleneck_features = x
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            x = self.ups[idx + 1](torch.cat((skip_connection, x), dim=1))

        return self.final_conv(x), bottleneck_features


class CounterNet(nn.Module):
    """
    Lightweight regression model to estimate:
    [house_count, tree_count] from the RGB patch.
    """

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


class SingleCounterNet(nn.Module):
    """
    Dedicated one-target regressor used separately for houses or trees.
    """

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


@dataclass(frozen=True)
class ClassInfo:
    index: int
    name: str
    rgb: Tuple[int, int, int]

    @property
    def pretty_name(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class SegmentationStats:
    counts: np.ndarray
    percentages: np.ndarray
    total_pixels: int
    dominant_idx: int
    present_indices: List[int]


@dataclass
class CountPrediction:
    house_count: Optional[int]
    tree_count: Optional[int]
    road_count: Optional[int] = None


@dataclass
class ClassMetrics:
    area_pixels: int
    area_percent: float
    region_count: int
    perimeter_pixels: float
    length_pixels: int


CLASS_ALIASES: Dict[str, List[str]] = {
    "urban_land": [
        "urban",
        "city",
        "built up",
        "builtup",
        "settlement",
        "road",
        "roads",
        "building",
        "buildings",
        "house",
        "houses",
        "home",
        "homes",
    ],
    "agriculture_land": ["agriculture", "agricultural", "farm", "farmland", "crop", "crops", "field", "fields"],
    "rangeland": ["range", "grassland", "open land", "pasture"],
    "forest_land": ["forest", "trees", "tree cover", "woodland"],
    "water": ["river", "lake", "pond", "sea", "ocean", "canal"],
    "barren_land": ["barren", "wasteland", "dry land", "empty land", "sand", "rocky"],
    "unknown": ["unknown", "unclassified", "other"],
}


LOCALIZATION_KEYWORDS: Tuple[str, ...] = (
    "point out",
    "locate",
    "where is",
    "where are",
    "highlight",
    "mark",
    "detect",
    "find",
    "show me",
)

COUNT_KEYWORDS: Tuple[str, ...] = (
    "how many",
    "count",
    "number of",
)

HOUSE_QUERY_KEYWORDS: Tuple[str, ...] = (
    "house",
    "houses",
    "home",
    "homes",
    "building",
    "buildings",
)

TREE_QUERY_KEYWORDS: Tuple[str, ...] = (
    "tree",
    "trees",
    "forest",
    "woodland",
    "teas",
)

ROAD_QUERY_KEYWORDS: Tuple[str, ...] = (
    "road",
    "roads",
    "street",
    "streets",
    "highway",
    "route",
)

LENGTH_QUERY_KEYWORDS: Tuple[str, ...] = (
    "length",
    "perimeter",
    "boundary",
    "outline",
)

AREA_QUERY_KEYWORDS: Tuple[str, ...] = (
    "area",
    "coverage",
    "percent",
    "percentage",
    "ratio",
)

ALL_LABELS_QUERY_KEYWORDS: Tuple[str, ...] = (
    "all labels",
    "all classes",
    "all metrics",
    "full report",
    "complete report",
)


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9_\- ]+", " ", text.lower()).strip()


def load_class_dict(csv_path: str) -> List[ClassInfo]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"class_dict not found: {csv_path}")

    classes: List[ClassInfo] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            classes.append(
                ClassInfo(
                    index=idx,
                    name=row["name"].strip(),
                    rgb=(int(row["r"]), int(row["g"]), int(row["b"])),
                )
            )

    if not classes:
        raise ValueError(f"No classes found in: {csv_path}")

    return classes


def load_model(model_path: str, n_classes: int, device: str) -> UNet:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = UNet(n_classes=n_classes).to(device)
    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if isinstance(state, dict) and all(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    model.load_state_dict(state)
    model.eval()
    return model


def load_counter_model(counter_model_path: str, device: str) -> Optional[CounterNet]:
    if not counter_model_path:
        return None
    if not os.path.exists(counter_model_path):
        return None

    model = CounterNet().to(device)
    state = torch.load(counter_model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_single_counter_model(counter_model_path: str, device: str) -> Optional[SingleCounterNet]:
    if not counter_model_path:
        return None
    if not os.path.exists(counter_model_path):
        return None

    model = SingleCounterNet().to(device)
    state = torch.load(counter_model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def resolve_image_path(user_input: str, image_dir: str) -> str:
    candidate = user_input.strip().strip('"').strip("'")
    candidate = os.path.expanduser(candidate)

    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate

    local_candidate = os.path.abspath(candidate)
    if os.path.exists(local_candidate):
        return local_candidate

    from_image_dir = os.path.abspath(os.path.join(image_dir, candidate))
    if os.path.exists(from_image_dir):
        return from_image_dir

    raise FileNotFoundError(f"Image not found: {user_input}")


def random_image_from_dir(image_dir: str) -> str:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = [
        file_name
        for file_name in os.listdir(image_dir)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ]
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {image_dir}")

    return os.path.join(image_dir, random.choice(image_files))


def load_image_rgb(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def image_to_tensor(image_rgb: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def predict_mask(model: UNet, image_rgb: np.ndarray, device: str) -> np.ndarray:
    logits = model(image_to_tensor(image_rgb, device))[0]
    pred_idx = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    return pred_idx


@torch.no_grad()
def predict_counts(counter_model: Optional[CounterNet], image_rgb: np.ndarray, device: str) -> Optional[CountPrediction]:
    if counter_model is None:
        return None

    resized = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    raw = counter_model(tensor)
    non_negative = F.softplus(raw).squeeze(0).cpu().numpy()
    house_count = int(round(float(non_negative[0])))
    tree_count = int(round(float(non_negative[1])))
    return CountPrediction(house_count=max(house_count, 0), tree_count=max(tree_count, 0), road_count=None)


@torch.no_grad()
def predict_single_count(counter_model: Optional[SingleCounterNet], image_rgb: np.ndarray, device: str) -> Optional[int]:
    if counter_model is None:
        return None

    resized = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    raw = counter_model(tensor)
    non_negative = F.softplus(raw).squeeze(0).cpu().numpy()
    count = int(round(float(non_negative[0])))
    return max(count, 0)


def compute_stats(mask_idx: np.ndarray, n_classes: int) -> SegmentationStats:
    counts = np.bincount(mask_idx.reshape(-1), minlength=n_classes).astype(np.int64)
    total_pixels = int(mask_idx.size)
    percentages = (counts / total_pixels) * 100.0 if total_pixels else np.zeros(n_classes, dtype=np.float64)
    dominant_idx = int(np.argmax(counts)) if total_pixels else 0
    present_indices = [int(idx) for idx, count in enumerate(counts) if count > 0]

    return SegmentationStats(
        counts=counts,
        percentages=percentages,
        total_pixels=total_pixels,
        dominant_idx=dominant_idx,
        present_indices=present_indices,
    )


def find_class_mentions(question: str, classes: Sequence[ClassInfo]) -> List[int]:
    normalized = normalize_text(question).replace("_", " ")
    mentions: List[int] = []

    for class_info in classes:
        aliases = [class_info.name, class_info.pretty_name]
        aliases.extend(CLASS_ALIASES.get(class_info.name, []))

        for alias in aliases:
            alias_norm = normalize_text(alias).replace("_", " ")
            if alias_norm and re.search(rf"\b{re.escape(alias_norm)}\b", normalized):
                mentions.append(class_info.index)
                break

    return mentions


def format_class_line(class_info: ClassInfo, stats: SegmentationStats) -> str:
    count = int(stats.counts[class_info.index])
    pct = float(stats.percentages[class_info.index])
    return f"{class_info.pretty_name}: {pct:.2f}% ({count} pixels)"


def build_summary(stats: SegmentationStats, classes: Sequence[ClassInfo], top_k: int = 3) -> str:
    sorted_indices = np.argsort(stats.counts)[::-1]
    top_indices = [int(idx) for idx in sorted_indices if stats.counts[idx] > 0][:top_k]
    if not top_indices:
        return "No predicted classes found."

    top_parts = [f"{classes[idx].pretty_name} ({stats.percentages[idx]:.2f}%)" for idx in top_indices]
    dominant = classes[stats.dominant_idx].pretty_name
    return f"Dominant class: {dominant}. Top coverage: {', '.join(top_parts)}."


def build_stats_report(stats: SegmentationStats, classes: Sequence[ClassInfo]) -> str:
    sorted_indices = np.argsort(stats.counts)[::-1]
    lines = [f"Total pixels: {stats.total_pixels}", "Class coverage:"]
    for idx in sorted_indices:
        class_info = classes[int(idx)]
        lines.append(f"- {format_class_line(class_info, stats)}")
    return "\n".join(lines)


def binary_mask_for_class(mask_idx: np.ndarray, class_idx: int) -> np.ndarray:
    return ((mask_idx == class_idx).astype(np.uint8) * 255)


def estimate_perimeter_pixels(binary_mask_255: np.ndarray) -> float:
    if cv2.countNonZero(binary_mask_255) == 0:
        return 0.0
    contours, _ = cv2.findContours(binary_mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = 0.0
    for contour in contours:
        perimeter += float(cv2.arcLength(contour, True))
    return perimeter


def estimate_skeleton_length_pixels(binary_mask_255: np.ndarray) -> int:
    if cv2.countNonZero(binary_mask_255) == 0:
        return 0

    work = binary_mask_255.copy()
    skeleton = np.zeros_like(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break

    return int(cv2.countNonZero(skeleton))


def compute_class_metrics(mask_idx: np.ndarray, stats: SegmentationStats, class_idx: int) -> ClassMetrics:
    binary = binary_mask_for_class(mask_idx, class_idx)
    regions = extract_object_regions(mask_idx, class_idx, min_pixels=5)

    return ClassMetrics(
        area_pixels=int(stats.counts[class_idx]),
        area_percent=float(stats.percentages[class_idx]),
        region_count=int(len(regions)),
        perimeter_pixels=float(estimate_perimeter_pixels(binary)),
        length_pixels=int(estimate_skeleton_length_pixels(binary)),
    )


def metrics_line(class_info: ClassInfo, metrics: ClassMetrics) -> str:
    return (
        f"{class_info.pretty_name}: area={metrics.area_percent:.2f}% ({metrics.area_pixels}px), "
        f"regions={metrics.region_count}, perimeter={metrics.perimeter_pixels:.1f}px, "
        f"length~={metrics.length_pixels}px"
    )


def build_all_labels_metrics_report(mask_idx: np.ndarray, stats: SegmentationStats, classes: Sequence[ClassInfo]) -> str:
    sorted_indices = np.argsort(stats.counts)[::-1]
    lines = ["All label metrics:"]
    for idx in sorted_indices:
        class_idx = int(idx)
        class_info = classes[class_idx]
        metrics = compute_class_metrics(mask_idx, stats, class_idx)
        lines.append(f"- {metrics_line(class_info, metrics)}")
    return "\n".join(lines)


def is_localization_request(question: str) -> bool:
    q = normalize_text(question).replace("_", " ")
    return any(keyword in q for keyword in LOCALIZATION_KEYWORDS)


def extract_object_regions(mask_idx: np.ndarray, class_idx: int, min_pixels: int = 20) -> List[Tuple[int, int, int, int, int, int, int, int]]:
    """
    Returns object-like connected regions for one class as:
    (region_id, x, y, w, h, area_px, centroid_x, centroid_y)
    """
    binary = (mask_idx == class_idx).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return []

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: List[Tuple[int, int, int, int, int, int, int, int]] = []

    for region_id in range(1, num_labels):
        x, y, w, h, area = stats[region_id]
        if int(area) < min_pixels:
            continue
        cx, cy = centroids[region_id]
        regions.append((region_id, int(x), int(y), int(w), int(h), int(area), int(round(cx)), int(round(cy))))

    regions.sort(key=lambda item: item[5], reverse=True)
    return regions


def save_localization_preview(
    image_rgb: np.ndarray,
    mask_idx: np.ndarray,
    class_info: ClassInfo,
    regions: Sequence[Tuple[int, int, int, int, int, int, int, int]],
    image_path: str,
    detections_dir: str,
    top_k: int = 12,
) -> str:
    os.makedirs(detections_dir, exist_ok=True)
    output = image_rgb.copy()

    class_mask = mask_idx == class_info.index
    color = np.array(class_info.rgb, dtype=np.float32)
    alpha = 0.35
    output[class_mask] = ((1.0 - alpha) * output[class_mask] + alpha * color).astype(np.uint8)

    draw_color = tuple(int(channel) for channel in class_info.rgb)
    for rank, (_, x, y, w, h, area, cx, cy) in enumerate(regions[:top_k], start=1):
        cv2.rectangle(output, (x, y), (x + w, y + h), draw_color, 2)
        cv2.circle(output, (cx, cy), 3, draw_color, -1)
        label = f"{class_info.pretty_name}#{rank} {area}px"
        cv2.putText(output, label, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1, cv2.LINE_AA)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    class_slug = class_info.name.lower().replace(" ", "_")
    output_name = f"{base_name}_{class_slug}_localized.png"
    output_path = os.path.abspath(os.path.join(detections_dir, output_name))

    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    return output_path


def localize_feature_response(
    question: str,
    image_rgb: np.ndarray,
    image_path: str,
    mask_idx: np.ndarray,
    stats: SegmentationStats,
    classes: Sequence[ClassInfo],
    detections_dir: str,
) -> str:
    mentions = find_class_mentions(question, classes)
    if not mentions:
        known = ", ".join(class_info.pretty_name for class_info in classes)
        return f"Please mention which feature to point out. Available classes: {known}."

    target_idx = mentions[0]
    class_info = classes[target_idx]

    if stats.counts[target_idx] == 0:
        return f"No {class_info.pretty_name} detected in this image."

    regions = extract_object_regions(mask_idx, target_idx)
    if not regions:
        regions = extract_object_regions(mask_idx, target_idx, min_pixels=1)
    if not regions:
        return f"{class_info.pretty_name} is present but no stable object regions were found."

    output_path = save_localization_preview(
        image_rgb=image_rgb,
        mask_idx=mask_idx,
        class_info=class_info,
        regions=regions,
        image_path=image_path,
        detections_dir=detections_dir,
    )

    largest = regions[0]
    _, x, y, w, h, area, cx, cy = largest
    return (
        f"Pointed out {class_info.pretty_name}: found {len(regions)} region(s). "
        f"Largest box -> x={x}, y={y}, w={w}, h={h}, area={area}px, center=({cx}, {cy}). "
        f"Saved: {output_path}"
    )


def answer_question(
    question: str,
    stats: SegmentationStats,
    classes: Sequence[ClassInfo],
    mask_idx: np.ndarray | None = None,
    count_prediction: Optional[CountPrediction] = None,
) -> str:
    q = normalize_text(question).replace("_", " ")
    mentions = find_class_mentions(question, classes)

    if any(token in q for token in ("help", "commands", "what can i ask")):
        return (
            "Ask about dominant class, class percentages, presence of a class, compare two classes, "
            "point out a class, class length/perimeter, road length, or all-label metrics. "
            "Examples: 'What is dominant?', 'How much water?', 'Is there forest?', "
            "'Compare water and urban land', 'Point out water', 'How many houses?', "
            "'Road length', 'All labels metrics'."
        )

    if mask_idx is not None and any(keyword in q for keyword in ALL_LABELS_QUERY_KEYWORDS):
        return build_all_labels_metrics_report(mask_idx, stats, classes)

    if any(token in q for token in ("dominant", "most", "majority", "largest", "main")):
        class_info = classes[stats.dominant_idx]
        return f"{class_info.pretty_name} is dominant with {stats.percentages[class_info.index]:.2f}% coverage."

    if any(token in q for token in ("least", "smallest", "minority")):
        present = [idx for idx in stats.present_indices if stats.counts[idx] > 0]
        if not present:
            return "No classes detected."
        least_idx = min(present, key=lambda idx: stats.counts[idx])
        class_info = classes[int(least_idx)]
        return f"{class_info.pretty_name} is the least present class at {stats.percentages[class_info.index]:.2f}%."

    if len(mentions) >= 2 and any(token in q for token in ("compare", "more", "less", "higher", "lower", "bigger")):
        first = classes[mentions[0]]
        second = classes[mentions[1]]
        first_pct = stats.percentages[first.index]
        second_pct = stats.percentages[second.index]
        diff = abs(first_pct - second_pct)

        if diff < 1e-6:
            return (
                f"{first.pretty_name} and {second.pretty_name} are equal at "
                f"{first_pct:.2f}% each."
            )

        if first_pct > second_pct:
            return (
                f"{first.pretty_name} is higher ({first_pct:.2f}%) than "
                f"{second.pretty_name} ({second_pct:.2f}%)."
            )

        return (
            f"{second.pretty_name} is higher ({second_pct:.2f}%) than "
            f"{first.pretty_name} ({first_pct:.2f}%)."
        )

    if mentions and mask_idx is not None and any(keyword in q for keyword in LENGTH_QUERY_KEYWORDS):
        target_idx = mentions[0]
        target_class = classes[target_idx]
        metrics = compute_class_metrics(mask_idx, stats, target_idx)
        is_road_query = any(keyword in q for keyword in ROAD_QUERY_KEYWORDS)

        if is_road_query and target_class.name == "urban_land":
            return (
                f"Approx road-length proxy from urban-land mask: {metrics.length_pixels}px. "
                f"Urban perimeter: {metrics.perimeter_pixels:.1f}px, regions: {metrics.region_count}."
            )

        return (
            f"{target_class.pretty_name} length~={metrics.length_pixels}px, "
            f"perimeter={metrics.perimeter_pixels:.1f}px, regions={metrics.region_count}."
        )

    if mentions and any(keyword in q for keyword in COUNT_KEYWORDS):
        if mask_idx is None:
            return "Counting requires the segmentation mask, which is not available right now."

        is_house_query = any(keyword in q for keyword in HOUSE_QUERY_KEYWORDS)
        is_tree_query = any(keyword in q for keyword in TREE_QUERY_KEYWORDS)
        is_road_query = any(keyword in q for keyword in ROAD_QUERY_KEYWORDS)

        house_counter_val = count_prediction.house_count if (count_prediction is not None and count_prediction.house_count is not None) else None
        tree_counter_val = count_prediction.tree_count if (count_prediction is not None and count_prediction.tree_count is not None) else None
        road_counter_val = count_prediction.road_count if (count_prediction is not None and count_prediction.road_count is not None) else None

        if is_house_query and is_tree_query:
            urban_idx = next((c.index for c in classes if c.name == "urban_land"), None)
            forest_idx = next((c.index for c in classes if c.name == "forest_land"), None)
            house_regions = len(extract_object_regions(mask_idx, urban_idx, min_pixels=6)) if urban_idx is not None else 0
            tree_regions = len(extract_object_regions(mask_idx, forest_idx, min_pixels=20)) if forest_idx is not None else 0
            house_blended = max(int(house_counter_val or 0), int(house_regions))
            tree_blended = max(int(tree_counter_val or 0), int(tree_regions))
            return (
                f"Estimated houses: {house_blended}. "
                f"Estimated trees: {tree_blended}. "
                f"(counter_h={house_counter_val}, blobs_h={house_regions}, "
                f"counter_t={tree_counter_val}, blobs_t={tree_regions})"
            )

        target_idx = mentions[0]
        target_class = classes[target_idx]

        min_pixels = 6 if is_house_query else 20
        regions = extract_object_regions(mask_idx, target_idx, min_pixels=min_pixels)

        if is_house_query and target_class.name == "urban_land":
            if house_counter_val is not None:
                blended = max(int(house_counter_val), int(len(regions)))
                return (
                    f"Estimated houses/buildings: {blended}. "
                    f"(counter={house_counter_val}, region_blobs={len(regions)})"
                )
            return (
                f"Estimated houses/building blobs: {len(regions)}. "
                "This is an approximation from urban-land segmentation regions, not true instance-level house detection."
            )

        if is_tree_query and target_class.name == "forest_land":
            if tree_counter_val is not None:
                blended = max(int(tree_counter_val), int(len(regions)))
                return (
                    f"Estimated trees: {blended}. "
                    f"(counter={tree_counter_val}, region_blobs={len(regions)})"
                )
            return (
                f"Estimated tree blobs: {len(regions)}. "
                "This is an approximation from forest-land segmentation regions."
            )

        if is_road_query and target_class.name == "urban_land":
            metrics = compute_class_metrics(mask_idx, stats, target_idx)
            if road_counter_val is not None:
                blended = max(int(road_counter_val), int(len(regions)))
                return (
                    f"Estimated road segments: {blended}. "
                    f"(counter={road_counter_val}, region_blobs={len(regions)}), "
                    f"road-length proxy={metrics.length_pixels}px."
                )
            return (
                f"Estimated road segments: {len(regions)}. "
                f"Road-length proxy={metrics.length_pixels}px."
            )

        return f"Estimated distinct {target_class.pretty_name} regions: {len(regions)}."

    if mentions:
        if mask_idx is not None and any(keyword in q for keyword in AREA_QUERY_KEYWORDS):
            target_idx = mentions[0]
            target_class = classes[target_idx]
            metrics = compute_class_metrics(mask_idx, stats, target_idx)
            return (
                f"{target_class.pretty_name} area={metrics.area_percent:.2f}% ({metrics.area_pixels}px), "
                f"regions={metrics.region_count}."
            )

        if any(token in q for token in ("is there", "do you see", "contains", "present", "any")):
            results = []
            for idx in mentions:
                class_info = classes[idx]
                has_class = stats.counts[idx] > 0
                verdict = "Yes" if has_class else "No"
                results.append(f"{verdict}, {class_info.pretty_name} ({stats.percentages[idx]:.2f}%).")
            return " ".join(results)

        if any(token in q for token in ("how much", "percent", "percentage", "area", "coverage", "ratio")):
            return " ".join(format_class_line(classes[idx], stats) + "." for idx in mentions)

        return " ".join(format_class_line(classes[idx], stats) + "." for idx in mentions)

    if any(token in q for token in ("which classes", "what classes", "class list", "present classes")):
        present_names = [classes[idx].pretty_name for idx in stats.present_indices if stats.counts[idx] > 0]
        if not present_names:
            return "No classes detected."
        return "Present classes: " + ", ".join(present_names) + "."

    return (
        build_summary(stats, classes)
        + " Ask about a specific class, road length, or use 'all labels metrics'."
    )


def run_chatbot(
    model: UNet,
    classes: Sequence[ClassInfo],
    image_dir: str,
    detections_dir: str,
    device: str,
    house_counter_model: Optional[SingleCounterNet] = None,
    tree_counter_model: Optional[SingleCounterNet] = None,
    road_counter_model: Optional[SingleCounterNet] = None,
    counter_model: Optional[CounterNet] = None,
) -> None:
    print(f"\nSegmentation VQA Bot ready on {device}.")
    print(
        "Counter models: "
        f"house={'yes' if house_counter_model is not None else 'no'}, "
        f"tree={'yes' if tree_counter_model is not None else 'no'}, "
        f"road={'yes' if road_counter_model is not None else 'no'}, "
        f"legacy_combined={'yes' if counter_model is not None else 'no'}."
    )
    print("Type an image path, or type 'random' to sample from your training image folder.")
    print("Type 'exit' anytime to quit.\n")

    while True:
        image_input = input("Image> ").strip()
        lower_input = image_input.lower()

        if lower_input in {"exit", "quit"}:
            print("Bye.")
            return

        try:
            image_path = random_image_from_dir(image_dir) if lower_input == "random" else resolve_image_path(image_input, image_dir)
            image_rgb = load_image_rgb(image_path)
            mask = predict_mask(model, image_rgb, device)
            stats = compute_stats(mask, n_classes=len(classes))

            house_count = predict_single_count(house_counter_model, image_rgb, device)
            tree_count = predict_single_count(tree_counter_model, image_rgb, device)
            road_count = predict_single_count(road_counter_model, image_rgb, device)

            combined_prediction = predict_counts(counter_model, image_rgb, device)
            if combined_prediction is not None:
                if house_count is None:
                    house_count = combined_prediction.house_count
                if tree_count is None:
                    tree_count = combined_prediction.tree_count

            count_prediction = CountPrediction(
                house_count=house_count,
                tree_count=tree_count,
                road_count=road_count,
            )
        except Exception as exc:
            print(f"Error: {exc}\n")
            continue

        print(f"\nLoaded image: {os.path.abspath(image_path)}")
        print(build_summary(stats, classes))
        print("Ask your question. Use 'stats' for full coverage, 'new' for another image, 'exit' to quit.\n")

        while True:
            question = input("You> ").strip()
            lower_question = question.lower()

            if not question:
                continue
            if lower_question in {"exit", "quit"}:
                print("Bye.")
                return
            if lower_question in {"new", "next", "change", "image"}:
                print("")
                break
            if lower_question == "stats":
                print(build_stats_report(stats, classes))
                print("")
                continue

            if is_localization_request(question):
                print(
                    "Bot>",
                    localize_feature_response(
                        question=question,
                        image_rgb=image_rgb,
                        image_path=image_path,
                        mask_idx=mask,
                        stats=stats,
                        classes=classes,
                        detections_dir=detections_dir,
                    ),
                )
                print("")
                continue

            print(
                "Bot>",
                answer_question(
                    question,
                    stats,
                    classes,
                    mask_idx=mask,
                    count_prediction=count_prediction,
                ),
            )
            print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal VQA chatbot for segmentation images.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to segmentation model weights (.pth).")
    parser.add_argument(
        "--house-counter-model-path",
        default=DEFAULT_HOUSE_COUNTER_MODEL_PATH,
        help="Path to trained house counter weights (.pth).",
    )
    parser.add_argument(
        "--tree-counter-model-path",
        default=DEFAULT_TREE_COUNTER_MODEL_PATH,
        help="Path to trained tree counter weights (.pth).",
    )
    parser.add_argument(
        "--road-counter-model-path",
        default=DEFAULT_ROAD_COUNTER_MODEL_PATH,
        help="Path to trained road counter weights (.pth).",
    )
    parser.add_argument(
        "--counter-model-path",
        default=DEFAULT_COUNTER_MODEL_PATH,
        help="Legacy combined house/tree counter weights (.pth).",
    )
    parser.add_argument("--class-dict", default=DEFAULT_CLASS_DICT_PATH, help="Path to class_dict.csv.")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Folder used by 'random' image selection.")
    parser.add_argument(
        "--detections-dir",
        default=DEFAULT_DETECTIONS_DIR,
        help="Folder where localized feature previews are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes = load_class_dict(args.class_dict)
    model = load_model(args.model_path, n_classes=len(classes), device=device)
    house_counter_model = load_single_counter_model(args.house_counter_model_path, device=device)
    tree_counter_model = load_single_counter_model(args.tree_counter_model_path, device=device)
    road_counter_model = load_single_counter_model(args.road_counter_model_path, device=device)
    counter_model = load_counter_model(args.counter_model_path, device=device)
    run_chatbot(
        model=model,
        classes=classes,
        image_dir=args.image_dir,
        detections_dir=args.detections_dir,
        device=device,
        house_counter_model=house_counter_model,
        tree_counter_model=tree_counter_model,
        road_counter_model=road_counter_model,
        counter_model=counter_model,
    )


if __name__ == "__main__":
    main()
