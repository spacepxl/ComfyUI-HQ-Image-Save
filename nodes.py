import os
import re
from glob import glob
from tqdm import tqdm, trange
import json

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
import torch
import numpy as np

import folder_paths
from comfy.cli_args import args
from comfy.utils import PROGRESS_BAR_ENABLED, ProgressBar


def sRGBtoLinear(npArray):
    less = npArray <= 0.0404482362771082
    npArray[less] = npArray[less] / 12.92
    npArray[~less] = np.power((npArray[~less] + 0.055) / 1.055, 2.4)

def linearToSRGB(npArray):
    less = npArray <= 0.0031308
    npArray[less] = npArray[less] * 12.92
    npArray[~less] = np.power(npArray[~less], 1/2.4) * 1.055 - 0.055

def load_EXR(filepath, tonemap):
    image = cv.imread(filepath, cv.IMREAD_UNCHANGED).astype(np.float32)
    if len(image.shape) == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=2)
    rgb = np.flip(image[:,:,:3], 2).copy()
    
    if tonemap == "sRGB":
        linearToSRGB(rgb)
        rgb = np.clip(rgb, 0, 1)
    elif tonemap == "Reinhard":
        rgb = np.clip(rgb, 0, None)
        rgb = rgb / (rgb + 1)
        linearToSRGB(rgb)
        rgb = np.clip(rgb, 0, 1)
    
    rgb = torch.unsqueeze(torch.from_numpy(rgb), 0)
    
    mask = torch.zeros((1, image.shape[0], image.shape[1]), dtype=torch.float32)
    if image.shape[2] > 3:
        mask[0] = torch.from_numpy(np.clip(image[:,:,3], 0, 1))
    
    return (rgb, mask)

def load_EXR_latent(filepath):
    image = cv.imread(filepath, cv.IMREAD_UNCHANGED).astype(np.float32)
    image = image[:,:, np.array([2,1,0,3])]
    image = torch.unsqueeze(torch.from_numpy(image), 0)
    image = torch.movedim(image, -1, 1)
    return (image)

def write_workflow(exr_path, prompt=None, extra_pnginfo=None):
    jsonpath = exr_path.rsplit(".", 1)[0]
    if prompt is not None:
        with open(jsonpath + "_api.json", "w") as f:
            f.write(json.dumps(prompt, indent=2))
    if extra_pnginfo is not None:
        with open(jsonpath + "_ui.json", "w") as f:
            for x in extra_pnginfo:
                f.write(json.dumps(extra_pnginfo[x], indent=2))

class LoadEXR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "path to directory or .exr file"}),
                "tonemap": (["linear", "sRGB", "Reinhard"], {"default": "sRGB"}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }

    CATEGORY = "HQ-Image-Save"

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("RGB", "alpha", "batch_size")
    FUNCTION = "load"

    def load(self, filepath, tonemap, image_load_cap, skip_first_images, select_every_nth):
        p = os.path.normpath(filepath.replace('\"', '').strip())
        if not os.path.exists(p):
            raise Exception("Path not found: " + p)
        
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() == ".exr":
            rgb, mask = load_EXR(p, tonemap)
            batch_size = 1
        else:
            rgb = []
            mask = []
            filelist = sorted(glob(os.path.join(p, "*.exr")))
            if not filelist:
                filelist = sorted(glob(os.path.join(p, "*.EXR")))
                if not filelist:
                    raise Exception("No EXRs found in folder")
            
            filelist = filelist[skip_first_images::select_every_nth]
            if image_load_cap > 0:
                cap = min(len(filelist), image_load_cap)
                filelist = filelist[:cap]
            batch_size = len(filelist)
            
            if PROGRESS_BAR_ENABLED:
                pbar = ProgressBar(batch_size)
            for file in tqdm(filelist, desc="loading images"):
                rgbFrame, maskFrame = load_EXR(file, tonemap)
                rgb.append(rgbFrame)
                mask.append(maskFrame)
                if PROGRESS_BAR_ENABLED:
                    pbar.update(1)
            
            rgb = torch.cat(rgb, 0)
            mask = torch.cat(mask, 0)
        
        return (rgb, mask, batch_size)

class LoadEXRFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "path/to/frame%04d.exr"}),
                "tonemap": (["linear", "sRGB", "Reinhard"], {"default": "sRGB"}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 9999}),
                "end_frame": ("INT", {"default": 1001, "min": 0, "max": 9999}),
            },
        }

    CATEGORY = "HQ-Image-Save"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("RGB", "alpha", "batch_size", "start_frame")
    FUNCTION = "load"

    def load(self, filepath, tonemap, start_frame, end_frame):
        if os.path.splitext(os.path.normpath(filepath))[1].lower() != ".exr":
            raise Exception("Filepath needs to end in .exr or .EXR")
        
        frames = list(range(start_frame, end_frame+1))
        if len(frames) == 0:
            raise Exception("Invalid frame range")
        
        if os.path.exists(os.path.normpath(filepath)): # absolute mode
            rgb, mask = load_EXR(os.path.normpath(filepath), tonemap)
            batch_size = 1
        elif "%04d" in filepath: # frame substitution
            rgb = []
            mask = []
            batch_size = len(frames)
            
            if PROGRESS_BAR_ENABLED and batch_size > 1:
                pbar = ProgressBar(batch_size)
            else:
                pbar = None
            
            for frame in tqdm(frames, desc="loading images"):
                framepath = os.path.normpath(filepath.replace("%04d", f"{frame:04}"))
                if os.path.exists(framepath):
                    rgbFrame, maskFrame = load_EXR(framepath, tonemap)
                    rgb.append(rgbFrame)
                    mask.append(maskFrame)
                else:
                    raise Exception("Frame not found: " + framepath)
                if pbar is not None:
                    pbar.update(1)
            rgb = torch.cat(rgb, 0)
            mask = torch.cat(mask, 0)
        else:
            raise Exception("Path not found: " + filepath)
        
        return (rgb, mask, batch_size, start_frame)

class SaveEXR:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "tonemap": (["linear", "sRGB", "Reinhard"], {"default": "sRGB"}),
                "version": ("INT", {"default": 1, "min": -1, "max": 999}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 99999999}),
                "frame_pad": ("INT", {"default": 4, "min": 1, "max": 8}),
                "save_workflow": (["ui", "api", "ui + api", "none"],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "HQ-Image-Save"

    def save_images(self, images, filename_prefix, tonemap, version, start_frame, frame_pad, save_workflow, prompt=None, extra_pnginfo=None):
        useabs = os.path.isabs(filename_prefix)
        if not useabs:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        
        linear = images.cpu().numpy().astype(np.float32)
        if tonemap != "linear":
            sRGBtoLinear(linear[...,:3]) # only convert RGB, not Alpha
        if tonemap == "Reinhard":
            linear[...,:3] = np.clip(linear[...,:3], 0, 0.999999)
            linear[...,:3] = -linear[...,:3] / (linear[...,:3] - 1)
        
        bgr = linear.copy()
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
        
        batch_size = linear.shape[0]
        if PROGRESS_BAR_ENABLED and batch_size > 1:
            pbar = ProgressBar(batch_size)
        else:
            pbar = None
        for i in trange(batch_size, desc="saving images"):
            if useabs:
                writepath = basepath + ver + f".{str(start_frame + i).zfill(frame_pad)}.exr"
            else:
                file = f"{filename}_{counter:05}_.exr"
                writepath = os.path.join(full_output_folder, file)
                counter += 1
            
            if os.path.exists(writepath):
                raise Exception("File exists already, stopping to avoid overwriting")
            cv.imwrite(writepath, bgr[i])
            
            if i < 1 and save_workflow != "none":
                api_json = prompt if "api" in save_workflow else None
                ui_json = extra_pnginfo if "ui" in save_workflow else None
                write_workflow(writepath, prompt=api_json, extra_pnginfo=ui_json)
            
            if pbar is not None:
                pbar.update(1)

        return { "ui": { "images": results } }

def safe_write_exr(writepath, overwrite, img):
    if os.path.exists(writepath):
        if overwrite:
            cv.imwrite(writepath, img)
        else:
            print(f"File {writepath} exists, skipping to avoid overwriting")
    else:
        cv.imwrite(writepath, img)

class SaveEXRFrames:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filepath": ("STRING", {"default": "path/to/frame%04d.exr"}),
                "tonemap": (["linear", "sRGB", "Reinhard"], {"default": "sRGB"}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 9999}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "save_workflow": (["ui", "api", "ui + api", "none"],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "HQ-Image-Save"

    def save_images(self, images, filepath, tonemap, start_frame, overwrite, save_workflow, prompt=None, extra_pnginfo=None):
        if os.path.splitext(os.path.normpath(filepath))[1].lower() != ".exr":
            raise Exception("Filepath needs to end in .exr or .EXR")
        
        if os.path.isabs(os.path.dirname(os.path.normpath(filepath))):
            os.makedirs(os.path.dirname(os.path.normpath(filepath)), exist_ok=True)
        else:
            raise Exception("Invalid filepath")
        
        results = list()
        
        linear = images.cpu().numpy().astype(np.float32)
        if tonemap != "linear":
            sRGBtoLinear(linear[...,:3]) # only convert RGB, not Alpha
        if tonemap == "Reinhard":
            linear[...,:3] = np.clip(linear[...,:3], 0, 0.999999)
            linear[...,:3] = -linear[...,:3] / (linear[...,:3] - 1)
        
        bgr = linear.copy()
        bgr[:,:,:,0] = linear[:,:,:,2] # flip RGB to BGR for opencv
        bgr[:,:,:,2] = linear[:,:,:,0]
        if bgr.shape[-1] > 3:
            bgr[:,:,:,3] = np.clip(1 - linear[:,:,:,3], 0, 1) # invert alpha
        
        api_json = prompt if "api" in save_workflow else None
        ui_json = extra_pnginfo if "ui" in save_workflow else None
        
        if "%04d" not in filepath: # write first frame only
            writepath = os.path.normpath(filepath)
            safe_write_exr(writepath, overwrite, bgr[0])
            
            if save_workflow != "none":
                write_workflow(writepath, prompt=api_json, extra_pnginfo=ui_json)
        else:
            batch_size = bgr.shape[0]
            
            if PROGRESS_BAR_ENABLED and batch_size > 1:
                pbar = ProgressBar(batch_size)
            else:
                pbar = None
            
            for i in trange(batch_size, desc="saving images"):
                writepath = os.path.normpath(filepath.replace("%04d", f"{start_frame + i:04}"))
                safe_write_exr(writepath, overwrite, bgr[i])
                
                if i < 1 and save_workflow != "none":
                    write_workflow(writepath, prompt=api_json, extra_pnginfo=ui_json)
                
                if pbar is not None:
                    pbar.update(1)

        return { "ui": { "images": results } }

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

    CATEGORY = "HQ-Image-Save"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        import imageio
        
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

class LoadLatentEXR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "path to directory or .exr file"}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }

    CATEGORY = "HQ-Image-Save"

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("samples", "batch_size")
    FUNCTION = "load"

    def load(self, filepath, image_load_cap=0, skip_first_images=0, select_every_nth=1):
        p = os.path.normpath(filepath.replace('\"', '').strip())
        if not os.path.exists(p):
            raise Exception("Path not found: " + p)
        
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() == ".exr":
            samples = load_EXR_latent(p)
            batch_size = 1
        else:
            samples = []
            filelist = sorted(glob(os.path.join(p, "*.exr")))
            if not filelist:
                filelist = sorted(glob(os.path.join(p, "*.EXR")))
                if not filelist:
                    raise Exception("No EXRs found in folder")
            
            filelist = filelist[skip_first_images::select_every_nth]
            if image_load_cap > 0:
                cap = min(len(filelist), image_load_cap)
                filelist = filelist[:cap]
            batch_size = len(filelist)
            
            if PROGRESS_BAR_ENABLED:
                pbar = ProgressBar(batch_size)
            for file in tqdm(filelist, desc="loading latents"):
                sampleFrame = load_EXR_latent(file)
                samples.append(sampleFrame)
                if PROGRESS_BAR_ENABLED:
                    pbar.update(1)
            
            samples = torch.cat(samples, 0)
        
        return ({"samples": samples}, batch_size)

class SaveLatentEXR:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        # self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
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

    CATEGORY = "HQ-Image-Save"

    def save_images(self, samples, filename_prefix, version, start_frame, frame_pad, prompt=None, extra_pnginfo=None):
        useabs = os.path.isabs(filename_prefix)
        linear = torch.movedim(samples["samples"], 1, -1)
        linear = linear.cpu().numpy().astype(np.float32)
        
        if not useabs:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, linear[0].shape[1], linear[0].shape[0])
        results = list()
        
        # flip rgb -> bgr for opencv
        linear = linear[:,:,:, np.array([2,1,0,3])]
        
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
        
        batch_size = linear.shape[0]
        if PROGRESS_BAR_ENABLED and batch_size > 1:
            pbar = ProgressBar(batch_size)
        else:
            pbar = None
        for i in trange(batch_size, desc="saving latents"):
            if useabs:
                writepath = basepath + ver + f".{str(start_frame + i).zfill(frame_pad)}.exr"
            else:
                file = f"{filename}_{counter:05}_.exr"
                writepath = os.path.join(full_output_folder, file)
                counter += 1
            
            if os.path.exists(writepath):
                raise Exception("File exists already, stopping to avoid overwriting")
            cv.imwrite(writepath, linear[i])
            if pbar is not None:
                pbar.update(1)

        return { "ui": { "images": results } }


class LoadImageAndPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "path to directory"}),
                "index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "prompt", "filename")
    FUNCTION = "load"

    CATEGORY = "HQ-Image-Save"
    
    def load(self, filepath, index):
        load_pattern = os.path.join(os.path.normpath(filepath), "*.*")
        all_files = glob(load_pattern)
        filtered_files = []
        for file in all_files:
            ext = os.path.splitext(file)[-1]
            if ext.lower() in [".jpg", ".jpeg", ".png", ".exr"]:
                filtered_files.append(file)
        
        image_filepath = filtered_files[index]
        text_filepath = os.path.splitext(image_filepath)[0] + ".txt"
        
        with open(text_filepath, "r") as prompt_file:
            text_prompt = prompt_file.read()
        
        image = cv.imread(image_filepath, cv.IMREAD_UNCHANGED)
        
        if len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        
        image = cv.cvtColor(image[..., :3], cv.COLOR_BGR2RGB)
        
        if image.dtype == np.float32:
            linearToSRGB(image)
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535
        
        image = np.clip(image, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        
        return (image, text_prompt, image_filepath)


class SaveImageAndPromptExact:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {
                    "default": "absolute filepath",
                    "tooltip": "The exact filepath to write the image to. Extension should be either .png or .exr, caption will be written to .txt",
                    }),
                "png_16bit": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
                "prompt": ("STRING", {"defaultInput": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "HQ-Image-Save"
    
    def save(self, filepath, png_16bit, image=None, alpha=None, prompt=None):
        assert image is not None or prompt is not None, "Must provide at least one of (image, prompt)"
        
        image_filepath = os.path.normpath(filepath)
        prompt_filepath = os.path.splitext(image_filepath)[0] + ".txt"
        image_ext = os.path.splitext(image_filepath)[-1]
        if image_ext.lower() in [".jpg", ".jpeg"]:
            image_filepath = os.path.splitext(image_filepath)[0] + ".png"
            image_ext = ".png"
        
        if image is not None:
            single_image = image[0].detach().clone()
            single_image = torch.flip(single_image, dims=[-1]) # BGR
            
            if alpha is not None:
                single_image = torch.cat([single_image, alpha[0].unsqueeze(-1)], dim=-1)
            
            single_image = single_image.float().cpu().numpy()
            
            if image_ext.lower() == ".png":
                if png_16bit:
                    single_image = (single_image * 65535).astype(np.uint16)
                else:
                    single_image = (single_image * 255).astype(np.uint8)
            elif image_ext.lower() == ".exr":
                sRGBtoLinear(single_image[..., :3])
            
            cv.imwrite(image_filepath, single_image)
        
        if prompt is not None:
            with open(prompt_filepath, "w") as prompt_file:
                prompt_file.write(prompt)
        
        return { "ui": { "images": list() } }


def get_highest_numbered_file(directory, prefix):
    pattern = os.path.join(directory, f"{prefix}*")
    files = glob(pattern)
    max_num = 0
    regex = re.compile(r'^' + re.escape(prefix) + r'(\d+).+$')
    
    if files:
        for file_path in files:
            filename = os.path.basename(file_path)
            match = regex.match(filename)
            
            if not match:
                continue # Skip files that don't match the expected pattern
            
            num_str = match.group(1)
            num = int(num_str)
            
            if num > max_num:
                max_num = num
    
    return max_num


class SaveImageAndPromptIncremental:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "folder path", "tooltip": "The folder to write in"}),
                "filename_prefix": ("STRING", {"default": ""}),
                "zero_padding": ("INT", {"default": 5}),
                "image_type": (["png", "png_16bit", "exr"], {"default": "png"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
                "prompt": ("STRING", {"defaultInput": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "HQ-Image-Save"
    
    def save(self, filepath, filename_prefix, zero_padding, image_type, image=None, alpha=None, prompt=None):
        assert image is not None or prompt is not None, "Must provide at least one of (image, prompt)"
        
        counter = get_highest_numbered_file(os.path.normpath(filepath), filename_prefix)
        batch_size = image.shape[0] if image is not None else 1
        pbar = ProgressBar(batch_size) if PROGRESS_BAR_ENABLED else None
        for i in trange(batch_size):
            counter += 1
            image_number = str(counter).zfill(zero_padding)
            file = os.path.join(os.path.normpath(filepath), filename_prefix + image_number)
            
            if image is not None:
                single_image = image[i].detach().clone()
                single_image = torch.flip(single_image, dims=[-1]) # BGR
                
                if alpha is not None:
                    single_image = torch.cat([single_image, alpha[i].unsqueeze(-1)], dim=-1)
                
                single_image = single_image.float().cpu().numpy()
                
                if image_type == "exr":
                    image_file = file + ".exr"
                    sRGBtoLinear(single_image[..., :3])
                else:
                    image_file = file + ".png"
                    if image_type == "png_16bit":
                        single_image = (single_image * 65535).astype(np.uint16)
                    else:
                        single_image = (single_image * 255).astype(np.uint8)
                
                cv.imwrite(image_file, single_image)
            
            if prompt is not None:
                with open(file + ".txt", "w") as prompt_file:
                    prompt_file.write(prompt)
            
            if pbar is not None:
                pbar.update(1)
        
        return { "ui": { "images": list() } }


NODE_CLASS_MAPPINGS = {
    "LoadEXR": LoadEXR,
    "LoadEXRFrames": LoadEXRFrames,
    "SaveEXR": SaveEXR,
    "SaveEXRFrames": SaveEXRFrames,
    "SaveTiff": SaveTiff,
    "LoadLatentEXR": LoadLatentEXR,
    "SaveLatentEXR": SaveLatentEXR,
    "LoadImageAndPrompt": LoadImageAndPrompt,
    "SaveImageAndPromptExact": SaveImageAndPromptExact,
    "SaveImageAndPromptIncremental": SaveImageAndPromptIncremental,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadEXR": "Load EXR",
    "LoadEXRFrames": "Load EXR Frames",
    "SaveEXR": "Save EXR",
    "SaveEXRFrames": "Save EXR Frames",
    "SaveTiff": "Save Tiff",
    "LoadLatentEXR": "Load Latent EXR",
    "SaveLatentEXR": "Save Latent EXR",
    "LoadImageAndPrompt": "Load Image And Prompt",
    "SaveImageAndPromptExact": "Save Image And Prompt (exact)",
    "SaveImageAndPromptIncremental": "Save Image And Prompt (incremental)",
}