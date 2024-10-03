# ComfyUI-HQ-Image-Save
## Nodes:
- Image
  - Load EXR (Individual file, or batch from folder, with cap/skip/nth controls in the same pattern as VHS load nodes)
  - Load EXR Frames (frame sequence with start/end frames, %04d frame formatting for filenames)
  - Save EXR (RGB or RGBA 32bpc EXR, with full support for batches and either relative paths in the output folder, or absolute paths with version and frame number formatting, and overwrite protection)
  - Save EXR Frames (frame sequence with %04d formatting, optionally save workflow as GUI .json and/or API .json)
  - Save Tiff (RGB 16bpc TIFF, outdated)
- Latent
  - Load Latent EXR (Same VHS style controls now)
  - Save Latent EXR (4 channel latent -> RGBA 32bpc EXR)

## Overview
Save and load images and latents as 32bit EXRs

I recommend adding the `--fp32-vae` CLI argument for more accurate decoding. If you get an error saying that the OpenEXR codec is disabled, see [this issue.](https://github.com/spacepxl/ComfyUI-HQ-Image-Save/issues/8)

Scatterplot of raw red/green values, left=PNG, right=EXR. PNG quantizes the image to 256 possible values per channel (2^8), while the EXR has full floating point precision.

![comparison](https://github.com/spacepxl/ComfyUI-HQ-Image-Save/assets/143970342/ce8107a2-31c9-44af-95af-b9ff8d704f7f)

For latent EXR viewing purposes, if you want a cheap approximation of RGB values from the four latent channels, use this formula:
```
r = (0.298 * r + 0.187 * g - 0.187 * b - 0.184 * a) * 0.18215
g = (0.207 * r + 0.286 * g + 0.189 * b - 0.271 * a) * 0.18215
b = (0.208 * r + 0.173 * g + 0.264 * b - 0.473 * a) * 0.18215
```

## Known Issues

- No load TIFF node, and the TIFF save is bad/outdated
