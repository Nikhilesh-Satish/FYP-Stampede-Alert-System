import argparse
from pathlib import Path

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create overlapping YOLO tiles for dense crowd training."
    )
    parser.add_argument("--data", required=True, help="Source dataset YAML path.")
    parser.add_argument(
        "--output",
        default="dataset_tiled",
        help="Output directory for tiled dataset.",
    )
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.35)
    parser.add_argument(
        "--min-visible",
        type=float,
        default=0.45,
        help="Minimum retained box area ratio after clipping.",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=6,
        help="Minimum width/height in pixels after clipping.",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def list_images(images_dir):
    return sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )


def read_labels(label_path, img_w, img_h):
    labels = []
    if not label_path.exists():
        return labels

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls_id, x_c, y_c, box_w, box_h = map(float, line.split())
        bw = box_w * img_w
        bh = box_h * img_h
        cx = x_c * img_w
        cy = y_c * img_h
        x1 = cx - (bw / 2.0)
        y1 = cy - (bh / 2.0)
        x2 = cx + (bw / 2.0)
        y2 = cy + (bh / 2.0)
        labels.append(
            {
                "cls": int(cls_id),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": max(0.0, bw) * max(0.0, bh),
            }
        )
    return labels


def tile_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def clip_box_to_tile(box, tile_x, tile_y, tile_size, min_visible, min_box_size):
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    if not (tile_x <= cx < tile_x + tile_size and tile_y <= cy < tile_y + tile_size):
        return None

    x1 = max(box["x1"], tile_x)
    y1 = max(box["y1"], tile_y)
    x2 = min(box["x2"], tile_x + tile_size)
    y2 = min(box["y2"], tile_y + tile_size)
    width = x2 - x1
    height = y2 - y1
    if width < min_box_size or height < min_box_size:
        return None

    retained = (width * height) / max(box["area"], 1.0)
    if retained < min_visible:
        return None

    return (
        box["cls"],
        ((x1 + x2) / 2.0 - tile_x) / tile_size,
        ((y1 + y2) / 2.0 - tile_y) / tile_size,
        width / tile_size,
        height / tile_size,
    )


def process_split(source_root, split_rel, output_root, tile_size, overlap, min_visible, min_box_size):
    images_dir = source_root / split_rel
    labels_dir = images_dir.parent / "labels"
    out_images = output_root / images_dir.parent.name / "images"
    out_labels = output_root / images_dir.parent.name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(tile_size * (1.0 - overlap)))
    written = 0

    for image_path in list_images(images_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        labels = read_labels(label_path, img_w, img_h)
        xs = tile_positions(img_w, tile_size, stride)
        ys = tile_positions(img_h, tile_size, stride)

        for tile_y in ys:
            for tile_x in xs:
                crop = image[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
                pad_h = tile_size - crop.shape[0]
                pad_w = tile_size - crop.shape[1]
                if pad_h > 0 or pad_w > 0:
                    crop = cv2.copyMakeBorder(
                        crop,
                        0,
                        pad_h,
                        0,
                        pad_w,
                        cv2.BORDER_CONSTANT,
                        value=(114, 114, 114),
                    )

                tile_labels = []
                for box in labels:
                    clipped = clip_box_to_tile(
                        box,
                        tile_x,
                        tile_y,
                        tile_size,
                        min_visible,
                        min_box_size,
                    )
                    if clipped:
                        tile_labels.append(clipped)

                if not tile_labels:
                    continue

                tile_name = f"{image_path.stem}_x{tile_x}_y{tile_y}"
                cv2.imwrite(str(out_images / f"{tile_name}.jpg"), crop)
                label_lines = [
                    f"{cls_id} {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}"
                    for cls_id, x_c, y_c, box_w, box_h in tile_labels
                ]
                (out_labels / f"{tile_name}.txt").write_text(
                    "\n".join(label_lines) + "\n",
                    encoding="utf-8",
                )
                written += 1

    return written


def main():
    args = parse_args()
    source_yaml = Path(args.data).resolve()
    source_cfg = load_yaml(source_yaml)
    source_root = Path(source_cfg["path"]).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": source_cfg.get("train"),
        "val": source_cfg.get("val"),
        "test": source_cfg.get("test"),
    }

    stats = {}
    for split_name, split_rel in split_map.items():
        if not split_rel:
            continue
        stats[split_name] = process_split(
            source_root,
            Path(split_rel),
            output_root,
            args.tile_size,
            args.overlap,
            args.min_visible,
            args.min_box_size,
        )

    tiled_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": source_cfg["names"],
    }
    save_yaml(output_root / "data.yaml", tiled_yaml)

    print("Tiled dataset written to:", output_root)
    for split_name, count in stats.items():
        print(f"{split_name}: {count} tiles")
    print("Dataset YAML:", output_root / "data.yaml")


if __name__ == "__main__":
    main()
