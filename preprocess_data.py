import os
import io
import json
import argparse
from itertools import islice

import torch
import webdataset as wds
import numpy as np
from PIL import Image

# Imagenette2 class mapping (folder name -> class index)
IMAGENETTE2_CLASSES = [
    "n01440764",  # tench
    "n02102040",  # English springer
    "n02979186",  # cassette player
    "n03000684",  # chain saw
    "n03028079",  # church
    "n03394916",  # French horn
    "n03417042",  # garbage truck
    "n03425413",  # gas pump
    "n03445777",  # golf ball
    "n03888257",  # parachute
]
IMAGENETTE2_CLASS_TO_IDX = {k: i for i, k in enumerate(IMAGENETTE2_CLASSES)}


def extract_class(meta: dict | None, key: str | None = None) -> int:
    """
    Prefer class from metadata if present, else infer from key.
    Imagenette2 keys commonly look like: "n01440764_12345" (depending on creation).
    """
    if isinstance(meta, dict):
        for candidate in ("cls", "class", "label", "synset"):
            v = meta.get(candidate)
            if isinstance(v, int):
                return int(v)
            if isinstance(v, str) and v in IMAGENETTE2_CLASS_TO_IDX:
                return IMAGENETTE2_CLASS_TO_IDX[v]

    if key:
        # Try to infer synset from the start of the key: "n01440764_..."
        syn = key.split("_", 1)[0]
        if syn in IMAGENETTE2_CLASS_TO_IDX:
            return IMAGENETTE2_CLASS_TO_IDX[syn]

    # Unknown -> -1 so you can filter later
    return -1


def augment(img_t: torch.Tensor) -> torch.Tensor:
    img_t = img_t.float()
    img_t = img_t + torch.randn_like(img_t) * 0.01
    img_t = img_t.clamp(0, 255)
    return img_t


def _to_png_bytes_hwc_uint8(img_hwc_uint8: np.ndarray) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(img_hwc_uint8).save(bio, format="PNG")
    return bio.getvalue()


def augment_wds(url: str, output: str, maxcount: int = 999999999) -> None:
    """
    Reads a WebDataset tar, decodes images, applies a tiny augmentation, and writes a new tar
    with:
      - __key__
      - png (encoded bytes)
      - cls (int label)
      - json (optional passthrough metadata)
    """
    src = (
        wds.WebDataset(url)
        .decode("torchrgb")  # yields CHW uint8 torch tensor
        .to_tuple("__key__", "jpg;png;jpeg", "json")
        .map_tuple(lambda x: x, augment, lambda j: j)
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with wds.TarWriter(output) as dst:
        for key, image_chw, meta in islice(src, 0, maxcount):
            if isinstance(meta, (bytes, bytearray)):
                try:
                    meta = json.loads(meta.decode("utf-8"))
                except Exception:
                    meta = None

            cls = extract_class(meta, key=key)

            # CHW uint8 -> HWC uint8 for PNG encoding
            image_hwc = image_chw.detach().cpu().numpy().transpose(1, 2, 0)
            image_hwc = np.clip(image_hwc, 0, 255).astype(np.uint8)
            png_bytes = _to_png_bytes_hwc_uint8(image_hwc)

            sample = {
                "__key__": key,
                "png": png_bytes,
                "cls": int(cls),
            }
            if meta is not None:
                sample["json"] = json.dumps(meta)

            dst.write(sample)


def _dir_to_wds(input_dir: str, output_tar: str, maxcount: int = 10**12) -> None:
    """
    Converts a flat ImageNet-style directory to a WebDataset tar.

    Expected layout (flat root):
      input_dir/<synset>/*.JPEG
      input_dir/<synset>/*.jpg
      input_dir/<synset>/*.png
    """
    root = input_dir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Expected directory: {root}")

    files: list[tuple[str, str]] = []
    for syn in sorted(os.listdir(root)):
        syn_dir = os.path.join(root, syn)
        if not os.path.isdir(syn_dir):
            continue
        for fn in os.listdir(syn_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append((syn, os.path.join(syn_dir, fn)))

    os.makedirs(os.path.dirname(os.path.abspath(output_tar)), exist_ok=True)
    with wds.TarWriter(output_tar) as dst:
        for i, (syn, path) in enumerate(files[:maxcount]):
            key = f"{syn}_{i:08d}"
            with open(path, "rb") as f:
                img_bytes = f.read()
            sample = {
                "__key__": key,
                "jpg": img_bytes,
                "json": json.dumps({"synset": syn, "path": os.path.relpath(path, input_dir)}),
            }
            dst.write(sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dir2tar", "augment"], required=True)
    ap.add_argument("--input", required=True, help="Flat root dir (dir2tar) or WDS url/path (augment)")
    ap.add_argument("--output", required=True, help="Output .tar path")
    ap.add_argument("--maxcount", type=int, default=999999999)
    args = ap.parse_args()

    if args.mode == "dir2tar":
        _dir_to_wds(args.input, args.output, maxcount=args.maxcount)
    else:
        augment_wds(args.input, args.output, maxcount=args.maxcount)


if __name__ == "__main__":
    main()