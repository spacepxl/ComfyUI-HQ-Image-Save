import torch
import os
import sys

#os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import imageio
import numpy as np
import copy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from comfy.cli_args import args

import folder_paths


class SaveTiff:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 65535. * image.cpu().numpy()
            img = np.clip(i, 0, 65535).astype(np.uint16)
            file = f"{filename}_{counter:05}_.tiff"
            imageio.imwrite(os.path.join(full_output_folder, file), img)
            #results.append({
            #    "filename": file,
            #    "subfolder": subfolder,
            #    "type": self.type
            #})
            counter += 1

        return { "ui": { "images": results } }

class SaveEXR:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = image.cpu().numpy()
            linear = copy.deepcopy(i)
            
            #sRGB -> linear conversion
            less = i <= 0.04045
            linear[less] = linear[less] / 12.92
            linear[~less] = np.power((linear[~less] + 0.055) / 1.055, 2.4)
            
            file = f"{filename}_{counter:05}_.exr"
            imageio.imwrite(os.path.join(full_output_folder, file), linear)
            #results.append({
            #    "filename": file,
            #    "subfolder": subfolder,
            #    "type": self.type
            #})
            counter += 1

        return { "ui": { "images": results } }

class SaveLatentEXR:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ),
                              "filename_prefix": ("STRING", {"default": "latents/ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ()
    FUNCTION = "save"

    OUTPUT_NODE = True

    CATEGORY = "latent"

    def save(self, samples, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        file = f"{filename}_{counter:05}_.exr"

        results = list()
        #results.append({
        #    "filename": file,
        #    "subfolder": subfolder,
        #    "type": "output"
        #})
        counter += 1
        
        file = os.path.join(full_output_folder, file)
        sample = torch.squeeze(samples["samples"], 0) # squeeze from [1, 4, x, y] to [4, x, y]
        output = torch.movedim(sample, 0, 2)          # and then reshape to [x, y, 4]
        imageio.imwrite(file, output)
        return { "ui": { "latents": results } }

class LoadLatentEXR:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".exr")]
        return {
                "required": {
                    "latent": (sorted(files), {"image_upload": True}),
                },
            }

    CATEGORY = "latent"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "load"

    def load(self, latent):
        latent_path = folder_paths.get_annotated_filepath(latent)
        read = imageio.imread(latent_path, flags=12) # freeimage FIT_RGBAF=12
        latent = torch.from_numpy(read).float()
        latent = torch.movedim(latent, 2, 0)         # reshape from [x, y, 4] to [4, x, y]
        latent = torch.unsqueeze(latent, 0)          # and then to [1, 4, x, y]
        samples = {"samples": latent}
        return (samples, )

    @classmethod
    def IS_CHANGED(s, latent):
        image_path = folder_paths.get_annotated_filepath(latent)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        if not folder_paths.exists_annotated_filepath(latent):
                return "Invalid latent file: {}".format(latent)
        return True

NODE_CLASS_MAPPINGS = {
    "SaveTiff": SaveTiff,
    "SaveEXR": SaveEXR,
    "SaveLatentEXR": SaveLatentEXR,
    "LoadLatentEXR": LoadLatentEXR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTiff": "Save Tiff",
    "SaveEXR": "Save EXR",
    "SaveLatentEXR": "Save Latent EXR",
    "LoadLatentEXR": "Load Latent EXR",
}
