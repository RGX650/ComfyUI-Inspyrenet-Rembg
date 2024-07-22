from .Inspyrenet_Rembg import (
    InspyrenetRembg,
    InspyrenetRembgAdvanced,
    InspyrenetRemover,
)

NODE_CLASS_MAPPINGS = {
    "InspyrenetRemover": InspyrenetRemover,
    "InspyrenetRembg": InspyrenetRembg,
    "InspyrenetRembgAdvanced": InspyrenetRembgAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspyrenetRemover": "Inspyrenet Remover",
    "InspyrenetRembg": "Inspyrenet Rembg",
    "InspyrenetRembgAdvanced": "Inspyrenet Rembg Advanced",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
