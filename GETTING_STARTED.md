## Getting Started with kNN-CLIP

Run all commands from the repository root. Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/main/GETTING_STARTED.md) for general usage.

Evaluation needs the [FC-CLIP ConvNeXt-Large checkpoint](https://drive.google.com/file/d/1-91PIns86vyNaL3CzMmDD39zKGnPMtvj/view?usp=sharing), which is the base model of the paper. The DINOv2 weights (4.5 GB) and the CLIP backbone download automatically on first use.

### Build a retrieval database

`build_database.py` makes a single pass over a set of images and their annotation maps, using one GPU. It writes `kNN.index` and `idx2class.npz` to `--out_dir`.

```bash
D=$DETECTRON2_DATASETS
python build_database.py \
  --img_dir $D/ADEChallengeData2016/images/training \
  --mask_dir $D/ADEChallengeData2016/annotations_detectron2/training \
  --out_dir retrieval_db/ade20k
```

Each benchmark has its own database, built from its training set. The default `--out_dir` below is what the matching config in [configs/coco/panoptic-segmentation/fcclip](configs/coco/panoptic-segmentation/fcclip) expects.

| Benchmark | Config `fcclip_convnext_large_eval_*.yaml` | `--img_dir` | `--mask_dir` | `--out_dir` |
|---|---|---|---|---|
| A-847 | `a847` | `$D/ADE20K_2021_17_01/images_detectron2/training` | `$D/ADE20K_2021_17_01/annotations_detectron2/training` | `retrieval_db/a847` |
| PC-459 | `pc459` | `$D/pascal_ctx_d2/images/training` | `$D/pascal_ctx_d2/annotations_ctx459/training` | `retrieval_db/pc459` |
| A-150 | `ade20k` | `$D/ADEChallengeData2016/images/training` | `$D/ADEChallengeData2016/annotations_detectron2/training` | `retrieval_db/ade20k` |
| PC-59 | `pc59` | `$D/pascal_ctx_d2/images/training` | `$D/pascal_ctx_d2/annotations_ctx59/training` | `retrieval_db/pc59` |
| PC-21 | `pas21` | `$D/pascal_voc_d2/images/training` | `$D/pascal_voc_d2/annotations_pascal21/training` | `retrieval_db/pas21` |
| COCO | `coco` | `$D/coco/train2017` | `$D/coco/panoptic_semseg_train2017` | `retrieval_db/coco` |

Database labels are the pixel values of the annotation maps, and they must index the vocabulary of the dataset being evaluated. A database therefore only works with its own benchmark.

A build processes about 3 images per second on an RTX A6000, which is under 2 hours for ADE20K and about 11 hours for COCO. Memory scales with the database: about 6 kB per embedding, for example about 1 GB for the 161k embeddings of A-150 and 5 GB for the 816k of COCO. Each evaluation process holds one copy in RAM.

### Evaluate

Set `OMP_NUM_THREADS` to your number of physical CPU cores first, or search can be about 20 times slower ([details](INSTALL.md#faiss-threads)).

```bash
export OMP_NUM_THREADS=4
python train_net.py --eval-only --num-gpus 1 \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml \
  MODEL.WEIGHTS /path/to/fcclip_cocopan.pth
```

Loading the checkpoint logs a warning that the `feature_extractor` weights are not found. This is expected, as DINOv2 loads its weights separately.

Options go after the other arguments as `KEY VALUE` pairs:

| Option | Default | Meaning |
|---|---|---|
| `MODEL.KNN_CLIP.ENABLED` | `True` | Use retrieval. `False` gives the plain FC-CLIP baseline. |
| `MODEL.KNN_CLIP.DATABASE` | set per config | Directory written by `build_database.py` |
| `MODEL.KNN_CLIP.NUM_NEIGHBOURS` | `16` | Retrieved neighbours $k$ |
| `MODEL.KNN_CLIP.CONF_THRESHOLD` | set per config | Confidence threshold $T$ |
| `MODEL.KNN_CLIP.LAMBDA` | `1.2` | Retrieval weight $\lambda$ |
| `MODEL.KNN_CLIP.DINOV2_WEIGHTS` | official weights | Local path or URL of the DINOv2 ViT-g/14 weights with registers |
| `MODEL.KNN_CLIP.IMAGE_SIZE` | `518` | DINOv2 input size. Must match `--image_size` of the database. |

The configs for PAS-20, Cityscapes and Mapillary Vistas run the FC-CLIP baseline, as the paper does not evaluate retrieval on them. To try it, build a database and add `MODEL.KNN_CLIP.ENABLED True MODEL.KNN_CLIP.DATABASE /path/to/database`.

### Approximate search

Add `--index_type hnsw` to build an HNSW index instead of the default exact search. `--ef_construction` and `--ef_search` default to the paper's 100 and 64. `--hnsw_m` defaults to 32. Evaluation picks up the index type from the file.

The index uses the inner product metric, as retrieval sums similarities. On A-150 with $T=0.75$, it gives the same accuracy as exact search (41.5 mIoU, 29.9 PQ and 17.4 AP, against 41.5, 30.0 and 17.4), and evaluation runs about 1.5 times faster. This differs from Table 7 of the paper, which reports 34.2 mIoU for HNSW on ADE20K.

### Grow a database

`--append` adds new images to an existing database in `--out_dir` without touching what it already holds, and needs none of the earlier images.

```bash
python build_database.py --append \
  --img_dir /path/to/new/images \
  --mask_dir /path/to/new/annotation_maps \
  --out_dir retrieval_db/ade20k
```

Annotation maps are `.png` (ignore value 255) or `.tif` (ignore value 65535) files named like their images, with one class id per pixel. To add a class that the vocabulary lacks, also add its name to the dataset's class list (the `*_with_prompt_eng.txt` files in [fcclip/data/datasets](fcclip/data/datasets)), so that the new id has text embeddings and a place in the model's output.

### Inference demo

```bash
python demo/demo.py \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml \
  --input input1.jpg input2.jpg \
  --output output_dir/ \
  --opts MODEL.WEIGHTS /path/to/fcclip_cocopan.pth
```

`--output` must be an existing directory when given several inputs. Without `--output`, results show in an OpenCV window, which needs the `opencv-python` package rather than the headless one. `--webcam` and `--video-input video.mp4` replace `--input`. Add `MODEL.DEVICE cpu` after `--opts` to run on CPU, which is slow.

With retrieval on, the demo uses the vocabulary of the config's test dataset, since the database labels index that vocabulary. With `MODEL.KNN_CLIP.ENABLED False`, it uses FC-CLIP's merged COCO, ADE20K and LVIS vocabulary, which you can extend through `user_classes` in [demo/predictor.py](demo/predictor.py).

### Train the base model

kNN-CLIP needs no training. To retrain FC-CLIP itself, drop `--eval-only`, which also turns retrieval off:

```bash
python train_net.py --num-gpus 8 \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml
```

The configs are made for 8-GPU training. On fewer GPUs, set `SOLVER.IMS_PER_BATCH` and `SOLVER.BASE_LR` to suit.
