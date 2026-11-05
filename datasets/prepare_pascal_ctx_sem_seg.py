"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/NVlabs/ODISE/blob/main/datasets/prepare_pascal_ctx_sem_seg.py

Modified by the kNN-CLIP authors: read the official .mat annotations and the
VOC2010 split lists, as trainval_merged.json is no longer available.
"""

import os
from pathlib import Path
import shutil

import numpy as np
import scipy.io as sio
import tqdm
from PIL import Image

# fmt: off
_mapping = np.sort(
    np.array([
        0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 23, 397, 25, 284,
        158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45, 46, 308, 59,
        440, 445, 31, 232, 65, 354, 424, 68, 326, 72, 458, 34, 207, 80, 355,
        85, 347, 220, 349, 360, 98, 187, 104, 105, 366, 189, 368, 113, 115
    ]))
# fmt: on
_key = np.array(range(len(_mapping))).astype("uint8")


def generate_labels(mat_file, out_file):
    label_map = sio.loadmat(mat_file)["LabelMap"]
    # classes outside the 59 are ignored, like the unlabelled class 0
    label_map = np.where(np.isin(label_map, _mapping), label_map, 0)
    index = np.digitize(label_map.ravel(), _mapping, right=True)
    sem_seg = _key[index].reshape(label_map.shape)
    sem_seg = sem_seg - 1  # 0 (ignore) becomes 255. others are shifted by 1

    Image.fromarray(sem_seg).save(out_file)


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "pascal_ctx_d2"
    voc_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "VOCdevkit/VOC2010"
    for split, split_list in [("training", "train"), ("validation", "val")]:
        image_ids = (voc_dir / "ImageSets" / "Main" / f"{split_list}.txt").read_text().split()

        output_img_dir = dataset_dir / "images" / split
        output_ann_dir = dataset_dir / "annotations_ctx59" / split

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_ann_dir.mkdir(parents=True, exist_ok=True)

        for image_id in tqdm.tqdm(image_ids, desc=f"Writing {split} to {dataset_dir} ..."):
            shutil.copy2(voc_dir / "JPEGImages" / f"{image_id}.jpg", output_img_dir)
            generate_labels(voc_dir / "trainval" / f"{image_id}.mat", output_ann_dir / f"{image_id}.png")
