# ComfyUI-HQ-Image-Save
## Nodes:
- Image
  - Save Tiff (RGB 16bpc TIFF, useful if you're applying a LUT or other color corrections, and care about preserving as much color accuracy as possible.)
  - Save EXR (RGB 32bpc EXR, mainly just for testing, it doesn't provide any improvement over TIFF in this situation, and it just takes up more file space.)
- Latent
  - Save Latent EXR (4 channel latent -> RGBA 32bpc EXR)
  - Load Latent EXR (images must be in the ComfyUI/input directory)

## Overview
Save images in TIFF 16 bit and EXR 32 bit formats, and save/load latent images as EXR

Don't expect any massive improvements in visible quality, this only passes through whatever is decoded by the VAE. From my testing there's an improvement in accuracy under the midpoint value per channel (0.5 raw, or ~0.215 gamma corrected), but still far less than what's actually possible within 16-bit depth.

![nodes](https://github.com/spacepxl/ComfyUI-HQ-Image-Save/assets/143970342/733fd3fa-ed85-430a-b772-3e84fede1675)


Here's an example scatterplot of raw R/G values from a generated image. You can see the greater variation in values below the 0.5 mark for each axis in the 16-bit version:

![comparison](https://github.com/spacepxl/ComfyUI-HQ-Image-Save/assets/143970342/0c12920a-5eae-4de2-83bc-3181fcbbbc64)

For latent EXR viewing purposes, if you want a cheap approximation of RGB values from the four latent channels, use this formula:
```
r = (0.298 * r + 0.187 * g - 0.187 * b - 0.184 * a) * 0.18215
g = (0.207 * r + 0.286 * g + 0.189 * b - 0.271 * a) * 0.18215
b = (0.208 * r + 0.173 * g + 0.264 * b - 0.473 * a) * 0.18215
```

## Installation
Navigate to `/ComfyUI/custom_nodes/`

`git clone https://github.com/spacepxl/ComfyUI-HQ-Image-Save/`

If you already have `imageio` installed, you can skip this, but otherwise, use the install.bat (shamelessly borrowed from WASasquatch) to automatically install dependencies, or manually run `pip install -r requirements.txt` using the correct python for your ComfyUI install (embedded or system)

If you want to use the EXR nodes, you'll need to add an environment variable to your run script to [enable EXR support](https://github.com/opencv/opencv/issues/21928)

For example on Windows, run_nvidia_gpu.bat:

```
set OPENCV_IO_ENABLE_OPENEXR=1
```

## Known Issues

- No metadata is saved to the images
- File Upload button is missing on the Load Latent EXR node
