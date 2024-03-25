# ComfyUI-HQ-Image-Save
## Nodes:
- Image
  - Load EXR (Individual file, or batch from folder, with cap/skip/nth controls in the same pattern as VHS load nodes)
  - Save EXR (RGB or RGBA 32bpc EXR, with full support for batches and either relative paths in the output folder, or absolute paths with version and frame number formatting, and overwrite protection)
  - Save Tiff (RGB 16bpc TIFF, needs update to match SaveEXR functionality)
- Latent
  - Load Latent EXR (Same VHS style controls now)
  - Save Latent EXR (4 channel latent -> RGBA 32bpc EXR)

## Overview
Save and load images and latents as 32bit EXRs

Recommend adding the `--fp32-vae` CLI argument for more accurate decoding.

Scatterplot of raw red/green values, left=PNG, right=TIFF. PNG quantizes the image to 256 possible values per channel (2^8), while the TIFF has 65,536 possible values per channel (2^16)

![comparison](https://github.com/spacepxl/ComfyUI-HQ-Image-Save/assets/143970342/ce8107a2-31c9-44af-95af-b9ff8d704f7f)

For latent EXR viewing purposes, if you want a cheap approximation of RGB values from the four latent channels, use this formula:
```
r = (0.298 * r + 0.187 * g - 0.187 * b - 0.184 * a) * 0.18215
g = (0.207 * r + 0.286 * g + 0.189 * b - 0.271 * a) * 0.18215
b = (0.208 * r + 0.173 * g + 0.264 * b - 0.473 * a) * 0.18215
```

## Known Issues

- No workflow metadata is saved to the TIFF or EXR images
- No load TIFF node yet
