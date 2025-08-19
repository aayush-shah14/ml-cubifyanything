# Cubify Anything

This repository includes the implementation of Cubify Transformer for CareMate for detecting 3D objects in an indoor scene environment and labeling via clip.

## Paper

**Apple**

[Cubify Anything: Scaling Indoor 3D Object Detection](https://arxiv.org/abs/2412.04458)

Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, Afshin Dehghan

![Teaser](teaser.jpg?raw=true "Teaser")

## Project Structure

```
cubifyanything/
├── cubifyanything/                  # Core library code
├── cubifyanything.egg-info/         # Package metadata
├── data/                            # Dataset directory (should contain a folder with video frames)
├── tools/                           # Utility scripts and tools
├── __pycache__/                     # Cached Python bytecode
├── customize_dataset.py             # Dataset custom script
├── object_room_map.json             # Mapping of objects to room categories
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup file
```

## Enhancements over the original pipeline

- New script demo_to_save.py takes in the video frame data formatted in ARKIT format and saves the predictions for each image in its .wide directory. The output will be stored in an instances.json format, with these fields:
  - **`id`** *(int or null)* — Unique identifier for the instance across frames if tracking is enabled; `null` when no persistent ID is assigned.
  - **`category`** *(str)* — High-level type of the prediction (e.g., `"object"`). Useful for grouping/filtering.
  - **`position`** *([x, y, z], meters)* — 3D center of the box in the **camera frame** for this image (right-handed; z points forward from the camera).
  - **`scale`** *([dx, dy, dz], meters)* — Physical box dimensions along the object’s **local** axes (width `dx`, height `dy`, depth `dz`). Half-extents are `scale/2`.
  - **`R`** *(3×3)* — Rotation matrix aligning the object’s **local** axes to the **camera** frame at this image (orthonormal; `det(R)=+1`). Apply as `X_cam = R @ X_local + position`.
  - **`corners`** *(8 × [x, y, z], meters)* — The 8 box corner coordinates in the **camera frame**, derived from `position`, `scale`, and `R`. Corner order is consistent but not semantically required (they collectively define the cuboid).
  - **`corners_world`** *(8 × [x, y, z], meters)* — The same 8 corners transformed into the **world frame** using the camera pose for this image.
  - **`box_2d_proj`** *([xmin, ymin, xmax, ymax], pixels)* — 2D axis-aligned bounding box from projecting `corners` with the camera intrinsics `K`; image-coordinate convention with origin at top-left.
  - **`box_2d_rend`** *([xmin, ymin, xmax, ymax], pixels)* — 2D box used for visualization/rendering (typically `box_2d_proj` after clamping to image bounds).
  - **`label`** *(str)* — Predicted class name (open-vocabulary text label, e.g., `"laptop"`).
  - **`confidence`** *(float, 0–1)* — Detector confidence used for thresholding/visualization (may differ from classifier score).
  - **`score`** *(float, 0–1)* — Class score/probability for the selected `label` (e.g., softmax/similarity-based).
  
- In addition to generating **CubifyAnything** predictions, the pipeline also performs **instance-level cropping** for each detected object. For every cropped instance, **semantic CLIP embeddings** are computed and used to assign **open-vocabulary labels** based on the `object_room_map.json` file.  
- The enhanced predictions (including 3D bounding boxes, labels, and confidence scores) are stored in `instances.json`, while cropped object images are saved in a dedicated output directory for further inspection or downstream tasks.

## Usage

1. **Download the pretrained model**  
   Place the `cutr_rgb.pth` model checkpoint in the project root directory.  

2. **Prepare the input video data**  
   Add real-world or simulated video sequences into the `data/` folder. Each frame must follow the ARKIT-style directory structure:  
   - `frame_timestamp.wide/` → contains `image.png` (the RGB image).  
   - `frame_timestamp.gt/` → contains `K.json` (intrinsics) and `RT.json` (extrinsics).  

3. **Run the demo script**  
   Use the following command to generate predictions, labels, and cropped outputs:  

   ```bash
   python tools/demo_to_save.py ./data/simulation-xxx --model-path ./cutr_rgb.pth --device mps
   ```

   - Set --device cuda when running on an NVIDIA GPU.

   - Replace simulation-xxx with the folder name of your dataset.
  
4. Optional arguments

   - --confidence-thres: Confidence threshold for CLIP labels during visualization.

   - --room_map_path: Path to the .json file containing object-to-room mappings (default: object_room_map.json).

   - --score-thresh: Minimum score threshold for Cubify Transformer predictions.


## Performance Metrics

The testing interface provides real-time monitoring of:
- **Latency**: End-to-end inference time
- **FPS**: Frames per second
- **GPU Memory**: Real-time GPU memory usage
- **Detection Count**: Number of objects detected

  
## Upcoming Features

- Evaluation utilities for quantitative assessment of prediction quality (e.g., IoU, precision/recall).
- Full CUDA GPU support for large-scale testing and improved runtime efficiency.
- Extended visualization tools for 3D rendering and scene-level analysis.
