from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm
import os

ckpt_path = os.getenv("INSPY_CKPT", "/workspace/ckpt/ckpt_base.pth")

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
            remover = Remover(ckpt=ckpt_path)
        else:
            remover = Remover(jit=True, ckpt=ckpt_path,)
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
