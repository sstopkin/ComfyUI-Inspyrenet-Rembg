import sys

# transparent_background's Remover optionally imports pymatting's cupy-accelerated
# foreground estimator. On some setups cupy is installed but its own CUDA-path
# autodetection is broken (seen in the wild: AttributeError: 'LoadedDL' object has
# no attribute 'found_via'), and that raises before Remover's own `except ImportError`
# fallback chain can catch it, crashing node execution outright. Pre-poison cupy in
# sys.modules so any subsequent `import cupy` fails as a clean ImportError instead,
# letting pymatting fall through to its CPU implementation like it's designed to.
try:
    import cupy  # noqa: F401
except Exception:
    sys.modules["cupy"] = None

from .Inspyrenet_Rembg import InspyrenetRembgAdvanced, TextOverlayConfig, TextOverlayBatch, VideoTextOverlay, PersonSelectionPreview

NODE_CLASS_MAPPINGS = {
    "InspyrenetRembgAdvanced": InspyrenetRembgAdvanced,
    "TextOverlayConfig": TextOverlayConfig,
    "TextOverlayBatch": TextOverlayBatch,
    "VideoTextOverlay": VideoTextOverlay,
    "PersonSelectionPreview": PersonSelectionPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspyrenetRembgAdvanced": "Inspyrenet Rembg Advanced",
    "TextOverlayConfig": "Text Overlay Config",
    "TextOverlayBatch": "Text Overlay Batch",
    "VideoTextOverlay": "Video Text Overlay",
    "PersonSelectionPreview": "Person Selection Preview",
}

__all__ = ['NODE_CLASS_MAPPINGS', "NODE_DISPLAY_NAME_MAPPINGS"]
