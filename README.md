# ComfyUI-HQ-Image-Save
Add Image Save nodes for TIFF 16 bit and EXR 32 bit formats. Probably only useful if you're applying a LUT or other color corrections, and care about preserving as much color accuracy as possible.

Don't expect any massive improvements in visible quality, this only passes through whatever is decoded by the VAE. From my testing there's an improvement in accuracy under the midpoint value per channel (0.5 raw, or ~0.215 gamma corrected), but still far less than what's actually possible within 16-bit depth.

Save EXR is mainly just for testing, it doesn't provide any improvement over TIFF in this situation, and it just takes up more file space.

![nodes](https://github.com/spacepxl/ComfyUI-HQ-Image-Save/assets/143970342/c385b4fc-e0cd-49e5-8679-fe7ce54854f3)

Here's an example scatterplot of raw R/G values from a generated image. You can see the tighter packing of values below the 0.5 mark for each axis in the 16-bit version:

![comparison](https://github.com/spacepxl/ComfyUI-HQ-Image-Save/assets/143970342/5838c51b-1308-41a0-9998-c7749c8e5dc4)


# Installation
Navigate to `/ComfyUI/custom_nodes/`

`git clone https://github.com/spacepxl/ComfyUI-HQ-Image-Save/`

If you already have `imageio` installed, you can skip this, but otherwise, use the install.bat (shamelessly borrowed from WASasquatch) to automatically install dependencies, or manually run `pip install -r requirements.txt` using the correct python for your ComfyUI install (embedded or system)

If you want to use Save EXR, you'll need to add an environment variable to your run script to [enable EXR support](https://github.com/opencv/opencv/issues/21928)

For example on Windows, run_nvidia_gpu.bat:

```
set OPENCV_IO_ENABLE_OPENEXR=1

.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --preview-method auto
pause
```

# Known Issues

Image previews don't display correctly in the nodes, although they do auto-expand as if there's an image to show.

No metadata is saved to the images
