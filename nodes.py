import os
import sys
import copy
import glob

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
import torch
import imageio
import numpy as np

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import folder_paths
from comfy.cli_args import args


def sRGBtoLinear(npArray):
    less = npArray <= 0.0404482362771082
    npArray[less] = npArray[less] / 12.92
    npArray[~less] = np.power((npArray[~less] + 0.055) / 1.055, 2.4)

def linearToSRGB(npArray):
    less = npArray <= 0.0031308
    npArray[less] = npArray[less] * 12.92
    npArray[~less] = np.power(npArray[~less], 1/2.4) * 1.055 - 0.055

def load_EXR(filepath, sRGB):
    image = cv.imread(filepath, cv.IMREAD_UNCHANGED).astype(np.float32)
    rgb = np.flip(image[:,:,:3], 2).copy()
    if sRGB:
        linearToSRGB(rgb)
        rgb = np.clip(rgb, 0, 1)
    rgb = torch.unsqueeze(torch.from_numpy(rgb), 0)
    
    mask = torch.zeros((1, image.shape[0], image.shape[1]), dtype=torch.float32)
    if image.shape[2] > 3:
        mask[0] = torch.from_numpy(np.clip(image[:,:,3], 0, 1))
    
    return (rgb, mask)

class LoadEXR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "path to directory or .exr file"}),
                "linear_to_sRGB": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("RGB", "alpha", "batch_size")
    FUNCTION = "load"

    def load(self, filepath, linear_to_sRGB=True, image_load_cap=0, skip_first_images=0, select_every_nth=1):
        p = os.path.normpath(filepath.replace('\"', '').strip())
        if not os.path.exists(p):
            raise Exception("Path not found: " + p)
        
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() == ".exr":
            rgb, mask = load_EXR(p, linear_to_sRGB)
            batch_size = 1
        else:
            rgb = []
            mask = []
            filelist = sorted(glob.glob(os.path.join(p, "*.exr")))
            if not filelist:
                filelist = sorted(glob.glob(os.path.join(p, "*.EXR")))
                if not filelist:
                    raise Exception("No EXRs found in folder")
            
            filelist = filelist[skip_first_images::select_every_nth]
            if image_load_cap > 0:
                cap = min(len(filelist), image_load_cap)
                filelist = filelist[:cap]
            batch_size = len(filelist)
            
            for file in filelist:
                rgbFrame, maskFrame = load_EXR(file, linear_to_sRGB)
                rgb.append(rgbFrame)
                mask.append(maskFrame)
            
            rgb = torch.cat(rgb, 0)
            mask = torch.cat(mask, 0)
        
        return (rgb, mask, batch_size)

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
        # self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "sRGB_to_linear": ("BOOLEAN", {"default": True}),
                "version": ("INT", {"default": 1, "min": -1, "max": 999}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 99999999}),
                "frame_pad": ("INT", {"default": 4, "min": 1, "max": 8}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, filename_prefix, sRGB_to_linear, version, start_frame, frame_pad, prompt=None, extra_pnginfo=None):
        useabs = os.path.isabs(filename_prefix)
        if not useabs:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        
        linear = images.detach().clone().cpu().numpy().astype(np.float32)
        if sRGB_to_linear:
            sRGBtoLinear(linear[:,:,:,:3]) # only convert RGB, not Alpha
        
        bgr = copy.deepcopy(linear)
        bgr[:,:,:,0] = linear[:,:,:,2] # flip RGB to BGR for opencv
        bgr[:,:,:,2] = linear[:,:,:,0]
        if bgr.shape[-1] > 3:
            bgr[:,:,:,3] = np.clip(1 - linear[:,:,:,3], 0, 1) # invert alpha
        
        if version < 0:
            ver = ""
        else:
            ver = f"_v{version:03}"
        
        if useabs:
            basepath = filename_prefix
            if os.path.basename(filename_prefix) == "":
                basename = os.path.basename(os.path.normpath(filename_prefix))
                basepath = os.path.join(os.path.normpath(filename_prefix) + ver, basename)
            if not os.path.exists(os.path.dirname(basepath)):
                os.mkdir(os.path.dirname(basepath))
        
        for i in range(linear.shape[0]):
            if useabs:
                writepath = basepath + ver + f".{str(start_frame + i).zfill(frame_pad)}.exr"
            else:
                file = f"{filename}_{counter:05}_.exr"
                writepath = os.path.join(full_output_folder, file)
                counter += 1
            
            if os.path.exists(writepath):
                raise Exception("File exists already, stopping to avoid overwriting")
            cv.imwrite(writepath, bgr[i])

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
    "LoadEXR": LoadEXR,
    "SaveTiff": SaveTiff,
    "SaveEXR": SaveEXR,
    "SaveLatentEXR": SaveLatentEXR,
    "LoadLatentEXR": LoadLatentEXR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadEXR": "Load EXR",
    "SaveTiff": "Save Tiff",
    "SaveEXR": "Save EXR",
    "SaveLatentEXR": "Save Latent EXR",
    "LoadLatentEXR": "Load Latent EXR",
}