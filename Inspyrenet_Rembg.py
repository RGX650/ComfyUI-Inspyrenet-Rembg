from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm
import os

from folder_paths import models_dir, add_model_folder_path, get_folder_paths

# KEEP Naming convention from https://github.com/plemeri/transparent-background/blob/main/transparent_background/Remover.py
# ckpt_dir\ckpt_name
# USE 'ComfyUI\models\transparent-background' like https://github.com/chflame163/ComfyUI_LayerStyle#TransparentBackgroundUltra

# Define the directory for Inspyrenet models
ckpt_dir = os.path.join(models_dir, "transparent-background")

# Ensure the Inspyrenet directory is registered in the paths
try:
    if ckpt_dir not in get_folder_paths("transparent-background"):
        raise KeyError
except KeyError:
    add_model_folder_path("transparent-background", ckpt_dir)

# Check if the inspyrinet_dir exists and is empty
if not os.path.exists(ckpt_dir) or not os.listdir(ckpt_dir):
    # Ensure the directory exists
    os.makedirs(ckpt_dir, exist_ok=True)

# use 
ckpt_name = os.path.join(ckpt_dir, "ckpt_base.pth")

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class InspyrenetRemover:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"torchscript_jit": (["default", "on"],)},
        }

    RETURN_TYPES = ("REMOVER",)
    FUNCTION = "init_remover"
    CATEGORY = "InspyreNet"

    def init_remover(self, torchscript_jit):
        if torchscript_jit == "default":
            remover = Remover(ckpt=ckpt_name)
        else:
            remover = Remover(jit=True, ckpt=ckpt_name,)
        return (remover,)


class InspyrenetRembg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"remover": ("REMOVER",), "image": ("IMAGE",),},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "InspyreNet"

    def remove_background(self, remover, image):
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type="rgba")
            out = pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        return (img_stack, mask)


class InspyrenetRembgAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "remover": ("REMOVER",),
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "InspyreNet"

    def remove_background(self, remover, image, threshold):
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type="rgba", threshold=threshold)
            out = pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        return (img_stack, mask)
