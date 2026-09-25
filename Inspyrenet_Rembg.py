from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm
import os
from typing import List, Tuple, Optional, Union
import logging
import gc
import folder_paths

# Registers a "transparent_background" models folder (models/transparent_background/)
# so alternate checkpoints (e.g. transparent-background's own "fast"/"base-nightly"
# variants) can be dropped in and picked from a dropdown instead of hardcoding a path.
# Idea from cfl-chenfangliang/ComfyUI-Inspyrenet-Rembg-opt.
if "transparent_background" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["transparent_background"] = (
        [os.path.join(folder_paths.models_dir, "transparent_background")],
        folder_paths.supported_pt_extensions,
    )


def get_available_models():
    """"(default)" first so the dropdown is never empty and never forces a
    choice — picking it just means "use Remover()'s own default resolution",
    same behaviour as before this option existed."""
    try:
        models = folder_paths.get_filename_list("transparent_background")
    except Exception:
        models = []
    return ["(default)"] + models

# List of available fonts - update this based on your font files
# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the fonts directory
fonts_directory = os.path.join(script_directory, "fonts")

AVAILABLE_FONTS = [f.split(".")[0] for f in os.listdir(
    fonts_directory) if f.endswith(".ttf") or f.endswith(".otf")]

# Remover() init is cheap relative to process() (~0.5s vs several seconds per
# image, measured), but still wasteful to pay on every single node call within
# the same ComfyUI session. Cache one instance per jit mode instead of creating
# a fresh one each time.
_remover_cache = {}


def get_cached_remover(jit, ckpt=None):
    key = (bool(jit), ckpt)
    if key not in _remover_cache:
        _remover_cache[key] = Remover(jit=key[0], ckpt=ckpt)
    return _remover_cache[key]


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def create_gradient_background(size, color1, color2, direction="horizontal"):
    """Create a gradient background."""
    width, height = size
    image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(image)

    if direction == "horizontal":
        for x in range(width):
            r = int(color1[0] + (float(x)/(width-1))*(color2[0]-color1[0]))
            g = int(color1[1] + (float(x)/(width-1))*(color2[1]-color1[1]))
            b = int(color1[2] + (float(x)/(width-1))*(color2[2]-color1[2]))
            draw.line([(x, 0), (x, height)], fill=(r, g, b, 255))
    else:  # vertical
        for y in range(height):
            r = int(color1[0] + (float(y)/(height-1))*(color2[0]-color1[0]))
            g = int(color1[1] + (float(y)/(height-1))*(color2[1]-color1[1]))
            b = int(color1[2] + (float(y)/(height-1))*(color2[2]-color1[2]))
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    return image


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def add_text_overlay(image, text_config):
    """Add text overlay to an image using the text configuration."""
    draw = ImageDraw.Draw(image)

    # Construct font paths based on font family for both ttf and otf
    font_path_ttf = os.path.join(
        fonts_directory, f"{text_config['font_family']}.ttf")
    font_path_otf = os.path.join(
        fonts_directory, f"{text_config['font_family']}.otf")

    # Try loading ttf first, then otf, and fallback to default if both fail
    if os.path.exists(font_path_ttf):
        font_path = font_path_ttf
    elif os.path.exists(font_path_otf):
        font_path = font_path_otf
    else:
        print(
            f"Error loading font {text_config['font_family']}, using default font")
        font_path = None
    try:
        font = ImageFont.truetype(font_path, text_config["font_size"])
    except Exception as e:
        print(
            f"Error loading font {text_config['font_family']}, using default font: {e}")
        font = ImageFont.load_default(text_config["font_size"])

    # Convert color if it's a hex string
    color = hex_to_rgb(text_config["text_color"])

    # Apply opacity
    color = (*color, int(255 * text_config["opacity"]))

    # Create a new image for rotated text if needed
    if text_config["rotation"] != 0:
        txt_img = Image.new('RGBA', image.size, (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text(text_config["position"], text_config["text_content"],
                      font=font, fill=color)
        rotated_txt = txt_img.rotate(text_config["rotation"], expand=True)
        image.paste(rotated_txt, (0, 0), rotated_txt)
    else:
        draw.text(text_config["position"], text_config["text_content"],
                  font=font, fill=color)

    return image


class TextOverlayConfig:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_content": ("STRING", {"default": "edit"}),
                "font_family": (AVAILABLE_FONTS,),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "text_placement": (["foreground", "background", "both"],),
                "x_position": ("FLOAT", {
                    "default": -12,
                    "min": -1000,
                    "max": 1000,
                    "step": 1
                }),
                "y_position": ("FLOAT", {
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "step": 1
                }),
                "font_size": ("INT", {
                    "default": 200,
                    "min": 10,
                    "max": 800,
                    "step": 1
                }),
                "font_weight": (["100", "200", "300", "400", "500", "600", "700", "800", "900"],),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "rotation": ("FLOAT", {
                    "default": 0,
                    "min": -360,
                    "max": 360,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("TEXT_CONFIG",)
    FUNCTION = "create_text_config"
    CATEGORY = "image/text"

    def create_text_config(self, text_content, font_family, text_color, text_placement,
                           x_position, y_position, font_size, font_weight, opacity, rotation):
        text_config = {
            "text_content": text_content,
            "font_family": font_family,
            "text_color": text_color,
            "text_placement": text_placement,
            "position": (x_position, y_position),
            "font_size": font_size,
            "font_weight": font_weight,
            "opacity": opacity,
            "rotation": rotation
        }
        return (text_config,)


class TextOverlayBatch:
    def __init__(self):
        self.text_configs = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_config_1": ("TEXT_CONFIG",),
            },
            "optional": {
                "text_config_2": ("TEXT_CONFIG",),
                "text_config_3": ("TEXT_CONFIG",),
                "text_config_4": ("TEXT_CONFIG",),
                "text_config_5": ("TEXT_CONFIG",),
            }
        }

    RETURN_TYPES = ("TEXT_CONFIG_BATCH",)
    FUNCTION = "batch_text_configs"
    CATEGORY = "image/text"

    def batch_text_configs(self, text_config_1, text_config_2=None, text_config_3=None,
                           text_config_4=None, text_config_5=None):
        self.text_configs.clear()
        self.text_configs.append(text_config_1)

        for config in [text_config_2, text_config_3, text_config_4, text_config_5]:
            if config is not None:
                self.text_configs.append(config)

        return (self.text_configs,)


class InspyrenetRembgAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torchscript_jit": (["default", "on"],),
                "output_type": (["original", "foreground", "background", "replace_background"],),
                "background_mode": (["color", "gradient", "image"],),
                "batch_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
            },
            "optional": {
                "background_color": ("STRING", {"default": "#000000"}),
                "gradient_color1": ("STRING", {"default": "#000000"}),
                "gradient_color2": ("STRING", {"default": "#FFFFFF"}),
                "gradient_direction": (["horizontal", "vertical"],),
                "background_image": ("IMAGE",),
                "text_config_batch": ("TEXT_CONFIG_BATCH",),
                "model": (get_available_models(), {"tooltip": "(default) uses Remover()'s own resolution. Otherwise pick a checkpoint placed in models/transparent_background/."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, threshold, torchscript_jit, output_type, background_mode,
                          batch_size=10, background_color="#000000", gradient_color1="#000000",
                          gradient_color2="#FFFFFF", gradient_direction="horizontal",
                          background_image=None, text_config_batch=None, model="(default)"):
        img_list = []
        mask_list = []

        # Initialize remover once for all batches
        remover = None
        if output_type != "original":
            ckpt_path = folder_paths.get_full_path_or_raise("transparent_background", model) if model != "(default)" else None
            remover = get_cached_remover(torchscript_jit == "on", ckpt=ckpt_path)
        
        total_images = len(image)
        
        # Process in batches
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            print(f"Processing batch {batch_start//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
            
            # Process current batch
            for img_idx in range(batch_start, batch_end):
                img = image[img_idx]
                pil_img = tensor2pil(img)
                original_size = pil_img.size

                if output_type == "original":
                    rgba_img = pil_img.convert('RGBA')
                    processed_output = rgba_img
                    alpha_mask = np.ones(
                        (original_size[1], original_size[0]), dtype=np.float32)
                else:
                    rgba_output = remover.process(
                        pil_img, type='rgba', threshold=threshold)
                    rgba_array = np.array(rgba_output)
                    alpha_mask = rgba_array[:, :, 3] / 255.0

                    if output_type == "replace_background":
                        if background_mode == "color":
                            bg_color = hex_to_rgb(background_color)
                            new_bg = Image.new(
                                'RGBA', original_size, (*bg_color, 255))
                        elif background_mode == "gradient":
                            color1 = hex_to_rgb(gradient_color1)
                            color2 = hex_to_rgb(gradient_color2)
                            new_bg = create_gradient_background(
                                original_size, color1, color2, gradient_direction)
                        elif background_mode == "image" and background_image is not None:
                            bg_idx = img_idx % len(background_image)
                            bg_pil = tensor2pil(background_image[bg_idx])
                            new_bg = bg_pil.resize(original_size).convert('RGBA')
                        else:
                            new_bg = Image.new(
                                'RGBA', original_size, (0, 0, 0, 255))

                        if text_config_batch:
                            for text_config in text_config_batch:
                                if text_config["text_placement"] in ["background", "both"]:
                                    new_bg = add_text_overlay(new_bg, text_config)

                        processed_output = Image.alpha_composite(
                            new_bg, rgba_output)

                        if text_config_batch:
                            for text_config in text_config_batch:
                                if text_config["text_placement"] in ["foreground", "both"]:
                                    processed_output = add_text_overlay(
                                        processed_output, text_config)

                    elif output_type == "background":
                        orig_rgba = np.concatenate(
                            [np.array(pil_img), np.ones_like(alpha_mask)[..., None] * 255], axis=-1)
                        inverse_mask = 1 - alpha_mask
                        bg_rgba = orig_rgba.copy()
                        bg_rgba[:, :, 3] = inverse_mask * 255
                        processed_output = Image.fromarray(
                            bg_rgba.astype(np.uint8))

                        if text_config_batch:
                            for text_config in text_config_batch:
                                if text_config["text_placement"] in ["background", "both"]:
                                    processed_output = add_text_overlay(
                                        processed_output, text_config)

                    else:  # foreground
                        processed_output = rgba_output
                        if text_config_batch:
                            for text_config in text_config_batch:
                                if text_config["text_placement"] in ["foreground", "both"]:
                                    processed_output = add_text_overlay(
                                        processed_output, text_config)

                out = pil2tensor(processed_output)
                img_list.append(out)

                mask_tensor = torch.from_numpy(alpha_mask).unsqueeze(0)
                mask_list.append(mask_tensor)
                
                # Clean up memory for this frame
                del pil_img, processed_output
                if 'rgba_output' in locals():
                    del rgba_output
                if 'new_bg' in locals():
                    del new_bg
            
            # Force garbage collection after each batch
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        img_stack = torch.cat(img_list, dim=0)
        mask_stack = torch.cat(mask_list, dim=0)

        if output_type == "background":
            mask_stack = 1 - mask_stack

        return (img_stack, mask_stack)


class VideoTextOverlay:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_type": (["original", "foreground", "background", "replace_background"],),
                                "background_mode": (["color", "gradient", "image"],),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torchscript_jit": (["default", "on"],),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "batch_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
            },
            "optional": {
                "background_color": ("STRING", {"default": "#000000"}),
                "gradient_color1": ("STRING", {"default": "#000000"}),
                "gradient_color2": ("STRING", {"default": "#FFFFFF"}),
                "gradient_direction": (["horizontal", "vertical"],),
                "background_images": ("IMAGE",),
                "text_config_batch": ("TEXT_CONFIG_BATCH",),
                "model": (get_available_models(), {"tooltip": "(default) uses Remover()'s own resolution. Otherwise pick a checkpoint placed in models/transparent_background/."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "image"

    def validate_images(self, pil_img: Image.Image, background_img: Optional[Image.Image] = None) -> bool:
        """Validate image sizes and formats"""
        if background_img is not None:
            if pil_img.size != background_img.size:
                self.logger.warning(
                    f"Size mismatch - Input: {pil_img.size}, Background: {background_img.size}")
                return False
        return True

    def process_frame(self, pil_frame: Image.Image, width: int, height: int,
                      output_type: str, background_mode: str,
                      remover, threshold: float, background_frame: Optional[Image.Image] = None,
                      background_color: str = "#000000",
                      gradient_color1: str = "#000000", gradient_color2: str = "#FFFFFF",
                      gradient_direction: str = "horizontal",
                      text_config_batch: Optional[List[dict]] = None) -> Image.Image:
        """Process a single frame with effects"""
        try:
            if output_type == "original":
                processed_frame = pil_frame.convert('RGBA')
            else:
                # Remove background
                try:
                    rgba_output = remover.process(
                        pil_frame, type='rgba', threshold=threshold)
                except Exception as e:
                    self.logger.error(f"Background removal failed: {str(e)}")
                    return pil_frame.convert('RGBA')

                if output_type == "replace_background":
                    # Validate and resize background if needed
                    if background_frame is not None:
                        if not self.validate_images(pil_frame, background_frame):
                            background_frame = background_frame.resize(
                                (width, height), Image.LANCZOS)
                            self.logger.info(
                                "Resized background image to match input dimensions")
                    try:
                        if background_mode == "color":
                            bg_color = hex_to_rgb(background_color)
                            new_bg = Image.new(
                                'RGBA', (width, height), (*bg_color, 255))
                        elif background_mode == "gradient":
                            color1 = hex_to_rgb(gradient_color1)
                            color2 = hex_to_rgb(gradient_color2)
                            new_bg = create_gradient_background(
                                (width, height), color1, color2, gradient_direction)
                        elif background_mode == "image" and background_frame is not None:
                            new_bg = background_frame.convert('RGBA')
                        else:
                            new_bg = Image.new(
                                'RGBA', (width, height), (0, 0, 0, 255))

                        # Apply background text overlays
                        if text_config_batch:
                            for text_config in text_config_batch:
                                if text_config["text_placement"] in ["background", "both"]:
                                    new_bg = add_text_overlay(
                                        new_bg, text_config)

                        processed_frame = Image.alpha_composite(
                            new_bg, rgba_output)
                    except Exception as e:
                        self.logger.error(
                            f"Background replacement failed: {str(e)}")
                        return pil_frame.convert('RGBA')

                elif output_type == "foreground":
                    processed_frame = rgba_output
                else:  # background
                    try:
                        rgba_array = np.array(rgba_output)
                        alpha_mask = rgba_array[:, :, 3] / 255.0
                        inverse_mask = 1 - alpha_mask

                        orig_rgba = np.concatenate([
                            np.array(pil_frame),
                            np.ones((height, width, 1), dtype=np.uint8) * 255
                        ], axis=-1)

                        bg_rgba = orig_rgba.copy()
                        bg_rgba[:, :, 3] = inverse_mask * 255
                        processed_frame = Image.fromarray(
                            bg_rgba.astype(np.uint8))
                    except Exception as e:
                        self.logger.error(
                            f"Background extraction failed: {str(e)}")
                        return pil_frame.convert('RGBA')

            # Apply foreground text overlays
            if text_config_batch:
                for text_config in text_config_batch:
                    if text_config["text_placement"] in ["foreground", "both"]:
                        processed_frame = add_text_overlay(
                            processed_frame, text_config)

            return processed_frame

        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            return pil_frame.convert('RGBA')

    def process_images(self, images: torch.Tensor, output_type: str,
                       background_mode: str, threshold: float,
                       torchscript_jit: str, fps: float,
                       batch_size: int = 10,
                       background_color: str = "#000000",
                       gradient_color1: str = "#000000",
                       gradient_color2: str = "#FFFFFF",
                       gradient_direction: str = "horizontal",
                       background_images: Optional[torch.Tensor] = None,
                       text_config_batch: Optional[List[dict]] = None,
                       model: str = "(default)") -> Tuple[torch.Tensor]:
        """Process batch of images with effects"""
        try:
            # Initialize the background remover once
            ckpt_path = folder_paths.get_full_path_or_raise("transparent_background", model) if model != "(default)" else None
            remover = get_cached_remover(torchscript_jit == "on", ckpt=ckpt_path)
            processed_frames = []
            
            total_frames = len(images)
            self.logger.info(f"Processing {total_frames} frames in batches of {batch_size}")
            
            # Process in batches
            for batch_start in range(0, total_frames, batch_size):
                batch_end = min(batch_start + batch_size, total_frames)
                batch_num = batch_start // batch_size + 1
                total_batches = (total_frames + batch_size - 1) // batch_size
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} (frames {batch_start}-{batch_end-1})")
                
                # Process frames in current batch
                batch_processed = []
                
                for idx in range(batch_start, batch_end):
                    try:
                        # Convert tensor to PIL Image
                        pil_img = tensor2pil(images[idx])
                        width, height = pil_img.size

                        # Get corresponding background image if available
                        bg_frame = None
                        if background_images is not None and len(background_images) > 0:
                            bg_idx = idx % len(background_images)
                            if idx % 50 == 0:  # Log every 50 frames to reduce log spam
                                self.logger.info(f"Frame {idx}: Using background image {bg_idx}")

                            if bg_idx < len(background_images):
                                bg_frame = tensor2pil(background_images[bg_idx])
                            else:
                                self.logger.warning(
                                    "Background image not found, using first background")
                                bg_frame = tensor2pil(background_images[0])

                        # Process the image
                        processed_frame = self.process_frame(
                            pil_img, width, height, output_type, background_mode,
                            remover, threshold, bg_frame, background_color,
                            gradient_color1, gradient_color2, gradient_direction,
                            text_config_batch
                        )

                        # Convert back to tensor and append to batch
                        batch_processed.append(pil2tensor(
                            processed_frame.convert('RGB')))
                        
                        # Clean up memory for this frame
                        del pil_img, processed_frame
                        if bg_frame is not None:
                            del bg_frame

                    except Exception as e:
                        self.logger.error(
                            f"Error processing frame {idx}: {str(e)}")
                        # Append original frame on error
                        batch_processed.append(images[idx])
                
                # Add batch to final list
                processed_frames.extend(batch_processed)
                
                # Clear batch from memory
                del batch_processed
                
                # Force garbage collection after each batch
                gc.collect()
                
                # Clear GPU/MPS cache if available
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Log memory usage (optional)
                if hasattr(torch.mps, 'driver_allocated_memory'):
                    allocated = torch.mps.driver_allocated_memory() / 1024**3
                    self.logger.info(f"MPS memory allocated: {allocated:.2f} GB")

            # Stack all processed frames into a single tensor
            self.logger.info("Stacking all processed frames...")
            output_tensor = torch.cat(processed_frames, dim=0)
            
            self.logger.info(f"Processing complete. Output shape: {output_tensor.shape}")
            return (output_tensor,)

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return (images,)  # Return original images on error


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TextOverlayConfig": TextOverlayConfig,
    "TextOverlayBatch": TextOverlayBatch,
    "InspyrenetRembgAdvanced": InspyrenetRembgAdvanced,
    "VideoTextOverlay": VideoTextOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextOverlayConfig": "Text Overlay Config",
    "TextOverlayBatch": "Text Overlay Batch",
    "InspyrenetRembgAdvanced": "Remove Background (Advanced)",
    "VideoTextOverlay": "Video Text Overlay",
}