from PIL import Image
from torchvision import transforms

from . import dinov2

DINOV2_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth"
DINOV2_FEAT_DIM = 1536

to_normalized_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def build_dinov2(weights=""):
    """Build the frozen DINOv2 ViT-g/14 with 4 registers used for retrieval."""
    extractor = dinov2.ViTFeat(
        weights or DINOV2_WEIGHTS_URL, DINOV2_FEAT_DIM, vit_feat="k"
    )
    extractor.eval()
    extractor.requires_grad_(False)
    return extractor


def preprocess_image(image, image_size):
    """Turn an RGB PIL image into a normalized [3, image_size, image_size] tensor."""
    image = image.resize((image_size, image_size), Image.LANCZOS)
    return to_normalized_tensor(image)


def extract_feature_maps(extractor, images):
    """Return the features of a preprocessed batch as [B, C, H, W] maps."""
    feats, _ = extractor(images)
    height, width = (
        size // extractor.patch_size for size in images.shape[-2:]
    )
    return feats.reshape(feats.shape[0], feats.shape[1], height, width)
